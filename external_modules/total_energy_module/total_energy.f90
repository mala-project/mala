SUBROUTINE initialize(y_planes_in, calculate_eigts_in)
  !----------------------------------------------------------------------------
  !  Derived from Quantum Espresso code
  !! author: Paolo Giannozzi
  !! license: GNU
  !! modified to enable calling Quantum Espresso functionality from Python
  !! modifications by Normand Modine, August 2020
  !
  !
  USE environment,       ONLY : environment_start
  USE mp_global,         ONLY : mp_startup
  USE mp,                ONLY : mp_size
  USE read_input,        ONLY : read_input_file
  USE command_line_options, ONLY: input_file_, command_line, ndiag_, nyfft_
  !
  IMPLICIT NONE
  CHARACTER(len=256) :: srvaddress
  !! Get the address of the server
  CHARACTER(len=256) :: get_server_address
  !! Get the address of the server
  INTEGER :: exit_status
  !! Status at exit
  LOGICAL :: use_images
  !! true if running "manypw.x"
  LOGICAL, EXTERNAL :: matches
  !! checks if first string is contained in the second
  !
  ! Optional arguments and defaults.
  LOGICAL, INTENT(IN), OPTIONAL :: calculate_eigts_in
  LOGICAL :: calculate_eigts = .false.
  INTEGER, INTENT(IN), OPTIONAL :: y_planes_in
    ! Parse optional arguments.
  IF (PRESENT(calculate_eigts_in)) THEN
    calculate_eigts = calculate_eigts_in
  ENDIF
  IF (PRESENT(y_planes_in)) THEN
    IF (y_planes_in > 1) THEN
      nyfft_ = y_planes_in
    ENDIF
  ENDIF

  !! checks if first string is contained in the second
  !
  CALL mp_startup ( start_images=.true., images_only=.true.)
  !
  CALL environment_start ( 'PWSCF' )
  !
  CALL read_input_file ('PW', 'mala.pw.scf.in' )
  CALL run_pwscf_setup ( exit_status, calculate_eigts)

  print *, "Setup completed"
  RETURN

END SUBROUTINE initialize
!
SUBROUTINE run_pwscf_setup ( exit_status, calculate_eigts)
  !----------------------------------------------------------------------------
  !  Derived from Quantum Espresso code
  !! author: Paolo Giannozzi
  !! license: GNU
  !! modified to enable calling Quantum Espresso functionality from Python
  !! modifications by Normand Modine, August 2020
  !
  !
  USE io_global,        ONLY : stdout, ionode, ionode_id
  USE parameters,       ONLY : ntypx, npk
  USE upf_params,       ONLY : lmaxx
  USE cell_base,        ONLY : fix_volume, fix_area
  USE control_flags,    ONLY : conv_elec, gamma_only, ethr, lscf
  USE control_flags,    ONLY : conv_ions, istep, nstep, restart, lmd, lbfgs, &
                               lforce => tprnfor
  USE command_line_options, ONLY : command_line
  USE force_mod,        ONLY : sigma, force
  USE check_stop,       ONLY : check_stop_init, check_stop_now
  USE mp_images,        ONLY : intra_image_comm
  USE extrapolation,    ONLY : update_file, update_pot
  USE scf,              ONLY : rho
  USE lsda_mod,         ONLY : nspin
  USE fft_base,         ONLY : dfftp
  USE qmmm,             ONLY : qmmm_initialization, qmmm_shutdown, &
                               qmmm_update_positions, qmmm_update_forces
  USE qexsd_module,     ONLY : qexsd_set_status
  USE xc_lib,           ONLY : xclib_dft_is, stop_exx
  !
  IMPLICIT NONE
  INTEGER, INTENT(OUT) :: exit_status
  !! Gives the exit status at the end
  LOGICAL, external :: matches
  !! checks if first string is contained in the second
  INTEGER :: idone
  !! counter of electronic + ionic steps done in this run
  INTEGER :: ions_status = 3
  !!    ions_status =  3  not yet converged
  !!    ions_status =  2  converged, restart with nonzero magnetization
  !!    ions_status =  1  converged, final step with current cell needed
  !!    ions_status =  0  converged, exiting
  !
  LOGICAL, INTENT(IN) :: calculate_eigts

  exit_status = 0
  IF ( ionode ) WRITE( unit = stdout, FMT = 9010 ) ntypx, npk, lmaxx
  !
  IF (ionode) CALL plugin_arguments()
  CALL plugin_arguments_bcast( ionode_id, intra_image_comm )
  !
  ! ... needs to come before iosys() so some input flags can be
  !     overridden without needing to write PWscf specific code.
  !
  CALL qmmm_initialization()
  !
  ! ... convert to internal variables
  !
  CALL iosys()
  !
  ! ... If executable names is "dist.x", compute atomic distances, angles,
  ! ... nearest neighbors, write them to file "dist.out", exit
  !
  IF ( matches('dist.x',command_line) ) THEN
     IF (ionode) CALL run_dist ( exit_status )
     RETURN
  END IF
  !
  IF ( gamma_only ) WRITE( UNIT = stdout, &
     & FMT = '(/,5X,"gamma-point specific algorithms are used")' )
  !
  ! call to void routine for user defined / plugin patches initializations
  !
  CALL plugin_initialization()
  !
  CALL check_stop_init()
  !
  CALL setup ()
  !
  CALL qmmm_update_positions()
  !
  ! ... dry run: code will stop here if called with exit file present
  ! ... useful for a quick and automated way to check input data
  !
  IF ( check_stop_now() ) THEN
     CALL pre_init()
     CALL data_structure( gamma_only )
     CALL summary()
     CALL memory_report()
     CALL qexsd_set_status(255)
     CALL punch( 'init-config' )
     exit_status = 255
     RETURN
  ENDIF
  !
  CALL init_run_setup(calculate_eigts)
  !
  IF ( check_stop_now() ) THEN
     CALL qexsd_set_status(255)
     CALL punch( 'config' )
     exit_status = 255
     RETURN
  ENDIF
  !
  RETURN
  !
9010 FORMAT( /,5X,'Current dimensions of program PWSCF are:', &
           & /,5X,'Max number of different atomic species (ntypx) = ',I2,&
           & /,5X,'Max number of k-points (npk) = ',I6,&
           & /,5X,'Max angular momentum in pseudopotentials (lmaxx) = ',i2)
  !
END SUBROUTINE run_pwscf_setup
!
SUBROUTINE init_run_setup(calculate_eigts)
  !----------------------------------------------------------------------------
  !  Derived from Quantum Espresso code
  !! author: Paolo Giannozzi
  !! license: GNU
  !! modified to enable calling Quantum Espresso functionality from Python
  !! modifications by Normand Modine, August 2020
  !
  USE kinds,              ONLY : DP
  USE klist,              ONLY : nkstot
  USE symme,              ONLY : sym_rho_init
  USE wvfct,              ONLY : nbnd, et, wg, btype
  USE control_flags,      ONLY : lmd, gamma_only, smallmem, ts_vdw
  USE gvect,              ONLY : g, gg, eigts1, eigts2, eigts3, mill, gcutm, ig_l2g, ngm, ngm_g, &
                                 gshells, gstart ! to be comunicated to the Solvers if gamma_only
  USE gvecs,              ONLY : gcutms, ngms
  USE cell_base,          ONLY : at, bg, set_h_ainv, alat, omega
  USE ions_base,          ONLY : zv, nat, nsp, ityp, tau
  USE cellmd,             ONLY : lmovecell
  USE dynamics_module,    ONLY : allocate_dyn_vars
  USE paw_variables,      ONLY : okpaw
  USE paw_init,           ONLY : paw_init_onecenter, allocate_paw_internals
  USE bp,                 ONLY : allocate_bp_efield, bp_global_map
  USE fft_base,           ONLY : dfftp, dffts
  USE xc_lib,             ONLY : xclib_dft_is_libxc, xclib_init_libxc, xclib_dft_is
  USE recvec_subs,        ONLY : ggen, ggens
  USE wannier_new,        ONLY : use_wannier    
  USE dfunct,             ONLY : newd
  USE esm,                ONLY : do_comp_esm, esm_init
  USE tsvdw_module,       ONLY : tsvdw_initialize
  USE Coul_cut_2D,        ONLY : do_cutoff_2D, cutoff_fact 
  USE ener,               ONLY : ewld
  USE vlocal,             ONLY : strf
  USE io_global, ONLY: stdout, ionode
  USE scf,       ONLY : rho_core, rhog_core
  USE mp,                ONLY : mp_size
  USE mp_world,          ONLY : world_comm



  IMPLICIT NONE
  LOGICAL, INTENT(IN) :: calculate_eigts

  REAL(DP), EXTERNAL :: ewald

  ! As per Paolo Giannozzis answer in the mailing list:
  ! "Once upon the time, the only limitation of 'memory="small"' was the
  ! impossibility to read data produced with N processors on M processors,
  ! if M != N (where N and M are actually "processors used for plane-wave
  ! parallelization"). This should still hold (no warranty)."
  ! there seems to be no problem with always having this true in our case.
  ! Setting it via MALA is also possible but a bit more tedious,
  ! since "memory" is not a supported option by e.g. ASE (and seems to
  ! be undocumented in the official documentation).
  CALL start_clock( 'init_run' )
  !
  ! ... calculate limits of some indices, used in subsequent allocations
  !
  CALL pre_init()
  !
  ! ... determine the data structure for fft arrays
  !
  CALL data_structure( gamma_only )
  !
  ! ... print a summary and a memory estimate before starting allocating
  !
  CALL summary()
  CALL memory_report()
  !
  ! ... allocate memory for G- and R-space fft arrays
  ! Note that this allocates a bunch of grid-sized quantities we don't use.
  ! We could save a modest amount of additional memory by pulling out only the
  ! variables we actually use.
  !
  CALL allocate_fft()
  !
  ! ... generate reciprocal-lattice vectors and fft indices
  !
  IF( mp_size(world_comm) > 1 ) THEN
     CALL ggen( dfftp, gamma_only, at, bg, gcutm, ngm_g, ngm, &
          g, gg, mill, ig_l2g, gstart, no_global_sort = .TRUE. )
  ELSE
     CALL ggen( dfftp, gamma_only, at, bg, gcutm, ngm_g, ngm, &
       g, gg, mill, ig_l2g, gstart )
  END IF
  !
  !  This seems to be needed by set_rhoc()
  !
  CALL gshells ( lmovecell )

  !
  ! ... allocate memory for structure factors
  !
  allocate (strf( ngm, nsp))

  IF (calculate_eigts) THEN
    allocate( eigts1(-dfftp%nr1:dfftp%nr1,nat) )
    allocate( eigts2(-dfftp%nr2:dfftp%nr2,nat) )
    allocate( eigts3(-dfftp%nr3:dfftp%nr3,nat) )
  END IF
  !  We do not initialize the structure factors, the core density, and the Ewald energy,
  !  since we do not intend to do a calculation for the atom positions in the dummy QE input file.
  !  This is consistent with not calculating the potentials and energies for the
  !  initial density,  However, the code will probably crash if results are requested
  !  before the positions and density are set.
  !
  CALL stop_clock( 'init_run' )
  !
  RETURN
  !
END SUBROUTINE init_run_setup
!
SUBROUTINE print_energies()
  !----------------------------------------------------------------------------
  !  Derived from Quantum Espresso code
  !! author: Paolo Giannozzi
  !! license: GNU
  !! modified to enable calling Quantum Espresso functionality from Python
  !! modifications by Normand Modine, August 2020
  !
  !
  USE ener, ONLY : eband, deband, ehart, etxc, etxcc, ewld
  USE io_global, ONLY: stdout, ionode

  IF ( ionode ) WRITE( unit = stdout, FMT = 9060 ) &
         deband, ehart, etxc, ewld
  !
  RETURN
  !
9060 FORMAT(/'     rho * v_hxc contribution  =',F17.8,' Ry' &
            /'     hartree contribution      =',F17.8,' Ry' &
            /'     xc contribution           =',F17.8,' Ry' &
            /'     ewald contribution        =',F17.8,' Ry' )

END SUBROUTINE print_energies

SUBROUTINE get_energies(e_rho_times_v_hxc,e_hartree,e_xc,e_ewald)
  USE ener, ONLY : deband, ehart, etxc, etxcc, ewld
  !----------------------------------------------------------------------------
  !  Derived from Quantum Espresso code
  !! author: Paolo Giannozzi
  !! license: GNU
  !! modified to enable calling Quantum Espresso functionality from Python
  !! modifications by Normand Modine, August 2020
  !
  !
  IMPLICIT NONE
  DOUBLE PRECISION, INTENT(OUT) :: e_rho_times_v_hxc
  DOUBLE PRECISION, INTENT(OUT) :: e_hartree
  DOUBLE PRECISION, INTENT(OUT) :: e_xc
  DOUBLE PRECISION, INTENT(OUT) :: e_ewald

  e_rho_times_v_hxc = deband
  e_hartree = ehart
  e_xc = etxc
  e_ewald = ewld
  !
  RETURN
  !
END SUBROUTINE get_energies


INTEGER FUNCTION get_nnr()
  !----------------------------------------------------------------------------
  !  Derived from Quantum Espresso code
  !! author: Paolo Giannozzi
  !! license: GNU
  !! modified to enable calling Quantum Espresso functionality from Python
  !! modifications by Normand Modine, August 2020
  !
  ! Get the value of dfftp%nnr, which is an argument to x_xc_wrapper()
  !
  USE fft_base,    ONLY : dfftp
  USE fft_types,   ONLY : fft_type_descriptor
  !
  TYPE(fft_type_descriptor) :: desc
  get_nnr = dfftp%nnr
END FUNCTION get_nnr


INTEGER FUNCTION get_proc_info()

  USE fft_base,    ONLY : dfftp
  USE fft_types,   ONLY : fft_type_descriptor
  !
  TYPE(fft_type_descriptor) :: desc
  INTEGER, ALLOCATABLE :: ir1p(:), ir1w(:)
  !! modifications by Jon Vogel, April 2022 for testing
  ! get the number of grid points per rank
  !get_nr1x = dfftp%nr1x
  !print *, "get_nr1x: ", get_nr1x
  !get_nr2x = dfftp%nr2x
  !print *, "get_nr2x: ", get_nr2x
  !get_nr3x = dfftp%nr3x
  !print *, "get_nr3x: ", get_nr3x
  ! get size of Y(2) and Z(3) section for each processor
  !get_mynr2p = dfftp%my_nr2p
  !print *, "get_mynr2p: ", get_mynr2p
  !get_mynr3p = dfftp%my_nr3p
  !print *, "get_mynr3p: ", get_mynr3p
  ! get offset of the 1st element for Y(2) and Z(3) on processor
  !get_myi0r2p = dfftp%my_i0r2p
  !print *, "get_my_i0r2p: ", get_myi0r2p
  !get_myi0r3p = dfftp%my_i0r3p
  !print *, "get_my_i0r3p: ", get_myi0r3p
  ! get total number of processors
  !get_nproc = dfftp%nproc
  !print *, "get_nproc: ", get_nproc
  ! get number of processors in the fft group along second dimension
  !get_nproc2 = dfftp%nproc2
  !print *, "get_nproc2: ", get_nproc2
  ! get number of processors in the fft group along third dimension
  !get_nproc3 = dfftp%nproc3
  !print *, "get_nproc3: ", get_nproc3
  ! get processor id for main fft communicator
  !get_mype = dfftp%mype
  !print *, "get_mype: ", get_mype
  ! get processor id for 2nd and 3rd dimensions
  !get_mype2 = dfftp%mype2
  !print *, "get_mype2: ", get_mype2
  !get_mype3 = dfftp%mype3
  !print *, "get_mype3: ", get_mype3
  ! print out the input fft grid dimensions
  !get_nr1 = dfftp%nr1
  !print *, "nr1: ", get_nr1
  !get_nr2 = dfftp%nr2
  !print *, "nr2: ", get_nr2
  !get_nr3 = dfftp%nr3
  !print *, "nr3: ", get_nr3

  !! work to match ijk index with real space fft grid points
  !open(171, file = 'ijk_parallelization')
  !j0 = dfftp%my_i0r2p ; k0 = dfftp%my_i0r3p
  !print *, j0, k0
  !DO ir = 1, dfftp%nr1x*dfftp%my_nr2p*dfftp%my_nr3p
  !   !
  !   ! ... three dimensional indexes
  !   !
  !   idx = ir -1
  !   k   = idx / (dfftp%nr1x*dfftp%my_nr2p)
  !   idx = idx - (dfftp%nr1x*dfftp%my_nr2p)*k
  !   k   = k + k0
  !   j   = idx / dfftp%nr1x
  !   idx = idx - dfftp%nr1x * j
  !   j   = j + j0
  !   i   = idx

  !   ! ... do not include points outside the physical range
  !   print *, i, j, k
  !   IF ( i >= dfftp%nr1 .OR. j >= dfftp%nr2 .OR. k >= dfftp%nr3 ) CYCLE
  !   !write(171,*) i, j, k
  !END DO

  !!get processor information and size distribution
  open(116, file='z_proc_info')
  write(116,*) "proc | proc z size | proc z offset | z element offset"
  DO lprocs = 1, dfftp%nproc3
   write(116,*) lprocs, dfftp%nr3p(lprocs), dfftp%nr3p_offset(lprocs), &
                dfftp%i0r3p(lprocs)
  END DO
  open(117, file='y_proc_info')
  write(117,*) "proc | proc y size | proc y offset | y element offset"
  DO lprocs = 1, dfftp%nproc2
   write(117,*) lprocs, dfftp%nr2p(lprocs), dfftp%nr2p_offset(lprocs), &
                dfftp%i0r2p(lprocs)
  END DO
  open(118, file='x_values_procs')
  write(118,*) "proc | active x values on proc: potential | wave"
  DO lprocs = 1, dfftp%nproc2
        write(118,*) lprocs, dfftp%nr1p(lprocs), dfftp%nr1w(lprocs)
  END DO

  !
END FUNCTION get_proc_info

INTEGER FUNCTION get_nspin()
  ! Get the value of lsda_mod nspin, which is an argument to x_xc_wrapper()
  USE lsda_mod,    ONLY : nspin
  get_nspin = nspin
END FUNCTION get_nspin

INTEGER FUNCTION get_ngm()
  ! Get the value of gvect ngm, which is an argument to x_xc_wrapper()
  USE gvect,        ONLY : ngm
  get_ngm = ngm
END FUNCTION get_ngm

INTEGER FUNCTION get_nat()
  ! Get the number of atoms, which is an argument to set_positions()
  USE ions_base,            ONLY : nat
  get_nat = nat
END FUNCTION get_nat

SUBROUTINE get_positions(positions,nat_in)
  !----------------------------------------------------------------------------
  !  Derived from Quantum Espresso code
  !! author: Paolo Giannozzi
  !! license: GNU
  !! modified to enable calling Quantum Espresso functionality from Python
  !! modifications by Normand Modine, August 2020
  !
  !
  !
  ! f2py gets confused when DP is used to define the type of an argument
  ! Instead, we use DOUBLE PRECISION for arguments
  !
  !USE kinds,       ONLY : DP
  USE scf,         ONLY : rho
  USE ions_base,   ONLY : nat, tau

  IMPLICIT NONE
  INTEGER,     INTENT(IN)  :: nat_in
  DOUBLE PRECISION,    INTENT(OUT)   :: positions(3,nat_in)

  ! Check consistency of dimensions
  IF (nat_in /= nat) STOP "*** nat provided to set_positions() does not match ions_base%nat"

  positions = tau

END SUBROUTINE get_positions


SUBROUTINE get_rho_of_r(rho_of_r,nnr_in,nspin_in)
  !----------------------------------------------------------------------------
  !  Derived from Quantum Espresso code
  !! author: Paolo Giannozzi
  !! license: GNU
  !! modified to enable calling Quantum Espresso functionality from Python
  !! modifications by Normand Modine, August 2020
  !
  !
  ! f2py gets confused when DP is used to define the type of an argument
  ! Instead, we use DOUBLE PRECISION for arguments
  !
  !USE kinds,       ONLY : DP
  USE scf,         ONLY : rho
  USE fft_base,    ONLY : dfftp
  USE lsda_mod,    ONLY : nspin

  IMPLICIT NONE
  INTEGER,     INTENT(IN)  :: nnr_in, nspin_in
  DOUBLE PRECISION,    INTENT(OUT)   :: rho_of_r(nnr_in,nspin_in)

  ! Check consistency of dimensions
  IF (nnr_in /= dfftp%nnr) STOP "*** nnr provided to set_rho_of_r() does not match dfftp%nnr"
  IF (nspin_in /= nspin) STOP "*** nspin provided to set_rho_of_r() does not mach lsda_mod%nspin"


  rho_of_r = rho%of_r

END SUBROUTINE get_rho_of_r

SUBROUTINE set_positions(positions, nat_in, skip_ewald_energy_in, skip_struct_fact_in)
  !  Derived from Quantum Espresso code
  !! author: Paolo Giannozzi
  !! license: GNU
  !! modified to enable calling Quantum Espresso functionality from Python
  !! modifications by Normand Modine, August 2020
  !
  !  Note that this assumes that the positions are in "crystal coordinates",
  !  i.e., that they are in units of the lattice vectors.
  !
  USE kinds,                ONLY : DP
  USE ions_base,            ONLY : nat, tau, nsp, ityp, zv
  USE gvect,                ONLY : ngm, g, gg, eigts1, eigts2, eigts3, gcutm, &
                                   gstart
  USE cell_base,            ONLY : omega, bg, alat, at
  USE fft_base,             ONLY : dfftp
  USE vlocal,               ONLY : strf
  USE control_flags,        ONLY : treinit_gvecs, gamma_only
  USE cellmd,               ONLY : lmovecell
  USE io_global,            ONLY : ionode_id, ionode, stdout
  USE mp_images,            ONLY : intra_image_comm
  USE symm_base,            ONLY : checkallsym
  USE mp,                   ONLY : mp_bcast
  USE ener,                 ONLY : ewld

  IMPLICIT NONE
  INTEGER,     INTENT(IN)  :: nat_in
  DOUBLE PRECISION,    INTENT(IN)   :: positions(3,nat_in)

  REAL(DP), EXTERNAL :: ewald
  LOGICAL, INTENT(IN), OPTIONAL :: skip_ewald_energy_in, skip_struct_fact_in
  LOGICAL :: skip_ewald_energy = .false.
  LOGICAL :: skip_struct_fact = .false.

  ! Optional arguments.

  IF (PRESENT(skip_ewald_energy_in)) THEN
    skip_ewald_energy = skip_ewald_energy_in
  ENDIF
  IF (PRESENT(skip_struct_fact_in)) THEN
    skip_struct_fact = skip_struct_fact_in
  ENDIF


  ! Check consistency of dimensions
  IF (nat_in /= nat) STOP "*** nat provided to set_positions() does not match ions_base%nat"

  !
  !  Update positions on one node
  !
  IF ( ionode ) THEN

     tau = positions

  END IF
  !
  !  Sync the positions across all processors
  !
  CALL mp_bcast( tau,       ionode_id, intra_image_comm )

  !
  !  Now, update everything else that depends on the postions
  !

  !
  !
  !  Calculate the structure factors for the new atom positions.
  !
  ! IF ( ionode ) WRITE( unit = stdout,FMT = *) 'Before struc_fact'
  IF (skip_struct_fact .eqv. .false.) THEN
    IF (ALLOCATED(eigts1)) THEN
      CALL struc_fact( nat, tau, nsp, ityp, ngm, g, bg, &
      dfftp%nr1, dfftp%nr2, dfftp%nr3, strf, eigts1, eigts2, eigts3 )
    ELSE
      CALL struc_fact_reduced( nat, tau, nsp, ityp, ngm, g, bg, &
      dfftp%nr1, dfftp%nr2, dfftp%nr3, strf)
    END IF
  ! IF ( ionode ) WRITE( unit = stdout,FMT = *) 'After struc_fact'
  !
  !  Update the core charge for the new positions.
  !
  ! IF ( ionode ) WRITE( unit = stdout,FMT = *) 'Before set_rhoc'
    CALL set_rhoc()
  END IF
  ! IF ( ionode ) WRITE( unit = stdout,FMT = *) 'After set_rhoc'

  !
  !  Update the ewald energy for the new positions
  !
  ! IF ( ionode ) WRITE( unit = stdout,FMT = *) 'Before ewald'
  IF (skip_ewald_energy .eqv. .false.) THEN
    ewld = ewald( alat, nat, nsp, ityp, zv, at, bg, tau, &
      omega, g, gg, ngm, gcutm, gstart, gamma_only, strf )
  ENDIF
  ! IF ( ionode ) WRITE( unit = stdout,FMT = *) 'After ewald'

  RETURN

END SUBROUTINE set_positions

SUBROUTINE set_positions_gauss(verbose, gaussian_descriptors,reference_gaussian_descriptors,sigma,nnr_in,nsp_in)
  !  Derived from Quantum Espresso code
  !! author: Paolo Giannozzi
  !! license: GNU
  !! modified to enable calling Quantum Espresso functionality from Python
  !! modifications by Normand Modine, August 2020
  !
  !  Note that this assumes that the positions are in "crystal coordinates",
  !  i.e., that they are in units of the lattice vectors.
  !
  USE kinds,                ONLY : DP
  USE ions_base,            ONLY : nsp
  USE gvect,                ONLY : ngm
  USE fft_base,             ONLY : dfftp
  USE vlocal,               ONLY : strf
  USE io_global,            ONLY : ionode, stdout
  USE mp_images,            ONLY : intra_image_comm
  USE mp,                   ONLY : mp_barrier
  USE fft_rho,              ONLY : rho_r2g

  IMPLICIT NONE
  INTEGER,     INTENT(IN)  :: nnr_in, nsp_in
  DOUBLE PRECISION,    INTENT(IN)   :: gaussian_descriptors(nnr_in,nsp_in)
  DOUBLE PRECISION,    INTENT(IN)   :: reference_gaussian_descriptors(nnr_in,1)
  DOUBLE PRECISION,    INTENT(IN)   :: sigma
  INTEGER,  INTENT(IN) :: verbose

  COMPLEX(DP), ALLOCATABLE :: rgd_of_g(:,:), rhon(:)
  INTEGER :: isp

  ! Check consistency of dimensions

  IF (nnr_in /= dfftp%nnr) STOP "*** nnr provided to set_positions() does not match dfftp%nnr"
  IF (nsp_in /= nsp) STOP "*** nsp provided to set_positions() does not match ions_base%nsp"

  !
  !  Compute the the structure factors
  !
  CALL mp_barrier( intra_image_comm )
  CALL start_clock( 'structure_factors' )
  ALLOCATE(rgd_of_g(ngm,1), rhon(ngm))

  CALL rho_r2g(dfftp, gaussian_descriptors, strf)

  CALL rho_r2g(dfftp, reference_gaussian_descriptors, rgd_of_g)

  DO isp = 1, nsp
     strf(:,isp) = strf(:,isp) / rgd_of_g(:,1)
  ENDDO
  CALL mp_barrier( intra_image_comm )
  CALL stop_clock( 'structure_factors')

  !
  !  Update the core charge for the new positions.
  !
  IF ( ionode .and. verbose > 1 ) WRITE( unit = stdout,FMT = *) 'Before set_rhoc'
  CALL mp_barrier( intra_image_comm )
  IF ( verbose > 1 ) CALL start_clock( 'set_rhoc' )
  CALL set_rhoc()
  CALL mp_barrier( intra_image_comm )
  IF (verbose > 1) THEN
    CALL stop_clock('set_rhoc')
    IF ( ionode ) WRITE( unit = stdout,FMT = *) 'After set_rhoc'
  END IF

  !
  !  Update the Ewald energy
  !
  IF ( ionode .and. verbose > 1 ) WRITE( unit = stdout,FMT = *) 'Before ewald'
  CALL mp_barrier( intra_image_comm )
  IF ( verbose > 1 ) CALL start_clock( 'set_ewald_lr')
  CALL set_ewald_lr( sigma )
  IF (verbose > 1) THEN
    CALL stop_clock( 'set_ewald_lr')
    CALL mp_barrier( intra_image_comm )
    IF ( ionode ) WRITE( unit = stdout,FMT = *) 'After ewald'
  END IF
  IF (verbose > 1 .and. ionode ) THEN
    CALL print_clock('structure_factors')
    CALL print_clock('set_rhoc')
    CALL print_clock('set_ewald_lr')
  END IF
  
  RETURN

END SUBROUTINE set_positions_gauss



!
!  Derived from Quantum Espresso code
!! author: Paolo Giannozzi
!! license: GNU
!! modified to enable calling structure factor calculation without 
!! calculation (and allocation) of eigts arrays
!! modifications by Lenz Fiedler, July 2022
!
!
!----------------------------------------------------------------------
subroutine struc_fact_reduced (nat, tau, ntyp, ityp, ngm, g, bg, nr1, nr2, &
  nr3, strf)
!----------------------------------------------------------------------
!
!   calculate the structure factors for each type of atoms in the unit
!   cell
!
USE kinds
USE constants, ONLY : tpi
implicit none
!
!   Here the dummy variables
!

integer :: nat, ntyp, ityp (nat), ngm, nr1, nr2, nr3
! input: the number of atom in the unit cel
! input: the number of atom types
! input: for each atom gives the type
! input: the number of G vectors
! input: fft dimension along x
! input: fft dimension along y
! input: fft dimension along z

real(DP) :: bg (3, 3), tau (3, nat), g (3, ngm)
! input: reciprocal crystal basis vectors
! input: the positions of the atoms in the c
! input: the coordinates of the g vectors

complex(DP) :: strf (ngm, ntyp)
! output: the structure factor
!
! output: the phases e^{-iG\tau_s}
!
!
!    here the local variables
!
integer :: nt, na, ng, n1, n2, n3, ipol
! counter over atom type
! counter over atoms
! counter over G vectors
! counter over fft dimension along x
! counter over fft dimension along y
! counter over fft dimension along z
! counter over polarizations

real(DP) :: arg, bgtau (3)
! the argument of the exponent
! scalar product of bg and tau

strf(:,:) = (0.d0,0.d0)
do nt = 1, ntyp
  do na = 1, nat
     if (ityp (na) .eq.nt) then
        do ng = 1, ngm
           arg = (g (1, ng) * tau (1, na) + g (2, ng) * tau (2, na) &
                + g (3, ng) * tau (3, na) ) * tpi
           strf (ng, nt) = strf (ng, nt) + CMPLX(cos (arg), -sin (arg),kind=DP)
        enddo
     endif
  enddo
enddo

return
end subroutine struc_fact_reduced



SUBROUTINE set_rho_of_r(rho_of_r,nnr_in,nspin_in)
  !----------------------------------------------------------------------------
  !  Derived from Quantum Espresso code
  !! author: Paolo Giannozzi
  !! license: GNU
  !! modified to enable calling Quantum Espresso functionality from Python
  !! modifications by Normand Modine, August 2020
  !
  !
  USE kinds,         ONLY : DP
  USE fft_base,      ONLY : dfftp
  USE scf,           ONLY : rho, rho_core, rhog_core, v
  USE ener,          ONLY : ehart, etxc, vtxc, deband
  USE ldaU,          ONLY : eth
  USE fft_rho,       ONLY : rho_r2g, rho_g2r
  USE lsda_mod,      ONLY : nspin
  USE cell_base,     ONLY : omega
  USE mp,            ONLY : mp_sum
  USE mp_bands,      ONLY : intra_bgrp_comm

  IMPLICIT NONE
  INTEGER,     INTENT(IN)  :: nnr_in, nspin_in
  DOUBLE PRECISION,    INTENT(IN)   :: rho_of_r(nnr_in,nspin_in)

  REAL(DP) :: charge
  REAL(DP) :: etotefield
  INTEGER :: ir

  ! Check consistency of dimensions
  IF (nnr_in /= dfftp%nnr) STOP "*** nnr provided to set_rho_of_r() does not match dfftp%nnr"
  IF (nspin_in /= nspin) STOP "*** nspin provided to set_rho_of_r() does not mach lsda_mod%nspin"

  !
  !  This calculates the G-space density from the input density.
  !
  CALL rho_r2g(dfftp, rho_of_r, rho%of_g)

  !
  !  The following results in a real-space density filtered at the plane-wave
  !  density cutoff.  This is needed to make it consistent with G-space density.
  !
  CALL rho_g2r(dfftp, rho%of_g, rho%of_r)


  !
  ! ... compute the potential and store it in v
  !
  CALL v_of_rho( rho, rho_core, rhog_core, &
                 ehart, etxc, vtxc, eth, etotefield, charge, v )

  !
  ! Compute:
  ! ... deband = - \sum_v <\psi_v | V_h + V_xc |\psi_v>
  ! Then:
  ! ... eband + deband = \sum_v <\psi_v | T + Vion |\psi_v>
  ! Note that the following is missing some terms from meta-GGA, LDA+U, and PAW
  ! See delta_e() in electrons.f90 for the full set of terms.
  deband = 0._dp
  IF ( nspin==2 ) THEN
     !
     DO ir = 1,dfftp%nnr
        deband = deband - ( rho%of_r(ir,1) + rho%of_r(ir,2) ) * v%of_r(ir,1) &  ! up
                        - ( rho%of_r(ir,1) - rho%of_r(ir,2) ) * v%of_r(ir,2)    ! dw
     ENDDO
     deband = 0.5_dp*deband
     !
  ELSE
     deband = - SUM( rho%of_r(:,:)*v%of_r(:,:) )
  ENDIF
  deband = omega * deband / ( dfftp%nr1*dfftp%nr2*dfftp%nr3 )
  !
  CALL mp_sum( deband, intra_bgrp_comm )
  !

  RETURN

END SUBROUTINE set_rho_of_r


SUBROUTINE v_xc_wrapper(rho_of_r, rho_of_g, rho_core, rhog_core,&
                         etxc, vtxc, v, nnr_in, nspin_in, ngm_in)
  !----------------------------------------------------------------------------
  !  Derived from Quantum Espresso code
  !! author: Paolo Giannozzi
  !! license: GNU
  !! modified to enable calling Quantum Espresso functionality from Python
  !! modifications by J. Adam Stephens and Normand Modine, 2020
  !
  ! rho_of_r and rho_of_g will be copied into arrays rho%of_r and rho%of_g.
  ! The arguments nnr_in, nspin_in, and ngm_in should be obtained by
  ! calling the functions get_nnr(), get_nspin(), and get_ngm(). Due to
  ! an apparent f2py limitation, it is not possible to declare the dimensions
  ! of the input arrays using module variables (e.g. dfftp%nnr). Other arguments
  ! are passed through unaltered to v_xc().
  USE gvect,       ONLY : ngm
  USE lsda_mod,    ONLY : nspin
  USE kinds,       ONLY : DP
  USE scf,         ONLY : scf_type
  USE fft_base,    ONLY : dfftp

  IMPLICIT NONE
  INTEGER,     INTENT(IN)  :: ngm_in, nnr_in, nspin_in
  DOUBLE PRECISION,    INTENT(IN)    :: rho_of_r(nnr_in,nspin_in), rho_core(nnr_in)
  DOUBLE COMPLEX, INTENT(IN)    :: rho_of_g(ngm_in,nspin_in), rhog_core(ngm_in)
  DOUBLE PRECISION,    INTENT(OUT)   :: v(nnr_in,nspin_in), vtxc, etxc

  ! local variables
  INTEGER :: allocate_status
  TYPE (scf_type) rho
  ! Check consistency of dimensions
  IF (nnr_in /= dfftp%nnr) STOP "*** nnr provided to v_xc_wrapper() does not match dfftp%nnr"
  IF (nspin_in /= nspin) STOP "*** nspin provided to v_xc_wrapper() does not mach lsda_mod%nspin"
  IF (ngm_in /= ngm) STOP "*** ngm provided to v_xc_wrapper() does not match gvect%ngm"
  ! Copy data to rho
  ALLOCATE ( rho%of_r(nspin, nspin), STAT = allocate_status)
  IF (allocate_status /= 0) STOP "*** Not enough memory ***"
  ALLOCATE ( rho%of_g(ngm, nspin), STAT = allocate_status)
  IF (allocate_status /= 0) STOP "*** Not enough memory ***"
  rho%of_r = rho_of_r
  rho%of_g = rho_of_g

  CALL v_xc( rho, rho_core, rhog_core, etxc, vtxc, v)

END SUBROUTINE v_xc_wrapper


SUBROUTINE v_h_wrapper(v, nnr_in, nspin_in)
  !----------------------------------------------------------------------------
  !  Derived from Quantum Espresso code
  !! author: Paolo Giannozzi
  !! license: GNU
  !! modified to enable calling Quantum Espresso functionality from Python
  !! modifications by J. Adam Stephens and Normand Modine, 2020
  !
  ! The arguments nnr_in, nspin_in, and ngm_in should be obtained by
  ! calling the functions get_nnr(), get_nspin(), and get_ngm(). Due to
  ! an apparent f2py limitation, it is not possible to declare the dimensions
  ! of the input arrays using module variables (e.g. dfftp%nnr). Other arguments
  ! are passed through unaltered to v_h().

  USE gvect,       ONLY : ngm
  USE lsda_mod,    ONLY : nspin
  USE kinds,       ONLY : DP
  USE fft_base,    ONLY : dfftp
!  USE fft_rho,       ONLY : rho_r2g
  USE scf,           ONLY : rho

  IMPLICIT NONE

  INTEGER,     INTENT(IN)  :: nnr_in, nspin_in
  DOUBLE PRECISION,  INTENT(INOUT) :: v(nnr_in,nspin_in)

  DOUBLE PRECISION :: ehart, charge

    ! Check consistency of dimensions
  IF (nnr_in /= dfftp%nnr) STOP "*** nnr provided to v_h_wrapper() does not match dfftp%nnr"
  IF (nspin_in /= nspin) STOP "*** nspin provided to v_h_wrapper() does not mach lsda_mod%nspin"

!  CALL rho_r2g(dfftp, rho_of_r, rho%of_g)
  CALL v_h(rho%of_g(:,1), ehart, charge, v)

END SUBROUTINE v_h_wrapper

!-----------------------------------------------------------------------
SUBROUTINE set_ewald_lr (sigma)
  !-----------------------------------------------------------------------
  !
  ! Calculates Ewald energy with both G- and R-space terms.
  ! Determines optimal alpha. Should hopefully work for any structure.
  !
  !
  USE kinds,                ONLY : DP
  USE constants,            ONLY : tpi, e2
  USE mp_bands,             ONLY : intra_bgrp_comm
  USE mp,                   ONLY : mp_sum
  USE ions_base,            ONLY : nat, nsp, ityp, zv
  USE gvect,                ONLY : ngm, gg, gstart
  USE cell_base,            ONLY : omega, alat
  USE vlocal,               ONLY : strf
  USE control_flags,        ONLY : gamma_only
  USE ener,                 ONLY : ewld

  implicit none

  DOUBLE PRECISION, INTENT(IN) :: sigma
  ! input: the Gaussian width used to compute the Ewald energy

  !    here the local variables
  !
  integer :: ng, na, nt
  ! counter over reciprocal G vectors
  ! counter on atoms
  ! counter on atomic types

  real(DP) :: charge, tpiba2, ewaldg, alpha, fact
  ! total ionic charge in the cell
  ! length in reciprocal space
  ! ewald energy computed in reciprocal space
  ! alpha term in ewald sum
  complex(DP) :: rhon

  alpha = 0.5d0/sigma**2

  tpiba2 = (tpi / alat) **2
  charge = 0.d0
  do na = 1, nat
     charge = charge+zv (ityp (na) )
  enddo
  !
  ! Only do th G-space sum
  !
  ! Determine if this processor contains G=0 and set the constant term
  !
  if (gstart==2) then
     ewaldg = - charge**2 / alpha / 4.0d0
  else
     ewaldg = 0.0d0
  endif
  if (gamma_only) then
     fact = 2.d0
  else
     fact = 1.d0
  end if
  do ng = gstart, ngm
     rhon = (0.d0, 0.d0)
     do nt = 1, nsp
        rhon = rhon + zv (nt) * CONJG(strf (ng, nt) )
     enddo
     ewaldg = ewaldg + fact * abs (rhon) **2 * exp ( - gg (ng) * tpiba2 / &
           alpha / 4.d0) / gg (ng) / tpiba2
  enddo
  ewaldg = 2.d0 * tpi / omega * ewaldg
  !
  !  Here add the other constant term
  !
  if (gstart.eq.2) then
     do na = 1, nat
        ewaldg = ewaldg - zv (ityp (na) ) **2 * sqrt (8.d0 / tpi * &
                alpha)
     enddo
  endif

  ewld = 0.5d0 * e2 * ewaldg

  call mp_sum(  ewld, intra_bgrp_comm )

END SUBROUTINE set_ewald_lr
