SUBROUTINE initialize()
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
  USE mp_world,          ONLY : world_comm
  USE mp_pools,          ONLY : intra_pool_comm
  USE mp_bands,          ONLY : intra_bgrp_comm, inter_bgrp_comm
  USE mp_diag,           ONLY : mp_start_diag
  USE mp_exx,            ONLY : negrp
  USE read_input,        ONLY : read_input_file
  USE command_line_options, ONLY: input_file_, command_line, ndiag_
  !
  IMPLICIT NONE
  CHARACTER(len=256) :: srvaddress
  !! Get the address of the server
  CHARACTER(len=256) :: get_server_address
  !! Get the address of the server
  INTEGER :: exit_status
  !! Status at exit
  LOGICAL :: use_images, do_diag_in_band_group = .true.
  !! true if running "manypw.x"
  LOGICAL, external :: matches
  !! checks if first string is contained in the second
  !
  CALL mp_startup ( start_images=.true.)
  !
  IF( negrp > 1 .OR. do_diag_in_band_group ) THEN
     ! used to be the default : one diag group per bgrp
     ! with strict hierarchy: POOL > BAND > DIAG
     ! if using exx groups from mp_exx still use this diag method
     CALL mp_start_diag ( ndiag_, world_comm, intra_bgrp_comm, &
          do_distr_diag_inside_bgrp_ = .true. )
  ELSE
     ! new default: one diag group per pool ( individual k-point level )
     ! with band group and diag group both being children of POOL comm
     CALL mp_start_diag ( ndiag_, world_comm, intra_pool_comm, &
          do_distr_diag_inside_bgrp_ = .false. )
  END IF
  CALL set_mpi_comm_4_solvers( intra_pool_comm, intra_bgrp_comm, &
       inter_bgrp_comm )
  !
  CALL environment_start ( 'PWSCF' )
  !
  CALL read_input_file ('PW', 'Al.scf.pw' )
  CALL run_pwscf_setup ( exit_status )

  print *, "Setup completed"

  RETURN

END SUBROUTINE initialize
!
SUBROUTINE run_pwscf_setup ( exit_status ) 
  !----------------------------------------------------------------------------
  !  Derived from Quantum Espresso code
  !! author: Paolo Giannozzi
  !! license: GNU 
  !! modified to enable calling Quantum Espresso functionality from Python
  !! modifications by Normand Modine, August 2020
  !
  !
  USE io_global,        ONLY : stdout, ionode, ionode_id
  USE parameters,       ONLY : ntypx, npk, lmaxx
  USE cell_base,        ONLY : fix_volume, fix_area
  USE control_flags,    ONLY : conv_elec, gamma_only, ethr, lscf
  USE control_flags,    ONLY : conv_ions, istep, nstep, restart, lmd, lbfgs
  USE command_line_options, ONLY : command_line
  USE force_mod,        ONLY : lforce, lstres, sigma, force
  USE check_stop,       ONLY : check_stop_init, check_stop_now
  USE mp_images,        ONLY : intra_image_comm
  USE extrapolation,    ONLY : update_file, update_pot
  USE scf,              ONLY : rho
  USE lsda_mod,         ONLY : nspin
  USE fft_base,         ONLY : dfftp
  USE qmmm,             ONLY : qmmm_initialization, qmmm_shutdown, &
                               qmmm_update_positions, qmmm_update_forces
  USE qexsd_module,     ONLY : qexsd_set_status
  USE funct,            ONLY : dft_is_hybrid, stop_exx 
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
  CALL init_run_setup()
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
SUBROUTINE init_run_setup()
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
  USE gvect,              ONLY : g, gg, mill, gcutm, ig_l2g, ngm, ngm_g, &
                                 gshells, gstart ! to be comunicated to the Solvers if gamma_only
  USE gvecs,              ONLY : gcutms, ngms
  USE cell_base,          ONLY : at, bg, set_h_ainv, alat, omega
  USE ions_base,          ONLY : zv, nat, nsp, ityp, tau
  USE cellmd,             ONLY : lmovecell
  USE dynamics_module,    ONLY : allocate_dyn_vars
  USE paw_variables,      ONLY : okpaw
  USE paw_init,           ONLY : paw_init_onecenter, allocate_paw_internals
#if defined(__MPI)
  USE paw_init,           ONLY : paw_post_init
#endif
  USE bp,                 ONLY : allocate_bp_efield, bp_global_map
  USE fft_base,           ONLY : dfftp, dffts
  USE funct,              ONLY : dft_is_hybrid
  USE recvec_subs,        ONLY : ggen, ggens
  USE wannier_new,        ONLY : use_wannier    
  USE dfunct,             ONLY : newd
  USE esm,                ONLY : do_comp_esm, esm_init
  USE tsvdw_module,       ONLY : tsvdw_initialize
  USE Coul_cut_2D,        ONLY : do_cutoff_2D, cutoff_fact 
  USE ener,               ONLY : ewld
  USE vlocal,             ONLY : strf

  IMPLICIT NONE

  REAL(DP), EXTERNAL :: ewald


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
  !
  CALL allocate_fft()
  !
  ! ... generate reciprocal-lattice vectors and fft indices
  !
  IF( smallmem ) THEN
     CALL ggen( dfftp, gamma_only, at, bg, gcutm, ngm_g, ngm, &
          g, gg, mill, ig_l2g, gstart, no_global_sort = .TRUE. )
  ELSE
     CALL ggen( dfftp, gamma_only, at, bg, gcutm, ngm_g, ngm, &
       g, gg, mill, ig_l2g, gstart )
  END IF
  CALL ggens( dffts, gamma_only, at, g, gg, mill, gcutms, ngms )
  if (gamma_only) THEN
     ! ... Solvers need to know gstart
     call export_gstart_2_solvers(gstart)
  END IF
  !
  IF (do_comp_esm) CALL esm_init()
  !
  ! ... setup the 2D cutoff factor
  !
  IF (do_cutoff_2D) CALL cutoff_fact()
  !
  CALL gshells ( lmovecell )
  !
  ! ... variable initialization for parallel symmetrization
  !
  CALL sym_rho_init (gamma_only )
  !
  ! ... allocate memory for all other arrays (potentials, wavefunctions etc)
  !
  CALL allocate_nlpot()
  IF (okpaw) THEN
     CALL allocate_paw_internals()
     CALL paw_init_onecenter()
  ENDIF
  CALL allocate_locpot()
  CALL allocate_bp_efield()
  CALL bp_global_map()
  !
  call plugin_initbase()
  !
  ALLOCATE( et( nbnd, nkstot ) , wg( nbnd, nkstot ), btype( nbnd, nkstot ) )
  !
  et(:,:) = 0.D0
  wg(:,:) = 0.D0
  !
  btype(:,:) = 1
  !
  IF (ts_vdw) THEN
     CALL tsvdw_initialize()
     CALL set_h_ainv()
  END IF
  !
  CALL openfil()
  !
  CALL hinit0()
  !
  CALL potinit()
  !
  CALL newd()
  !
  IF(use_wannier) CALL wannier_init()
  !
#if defined(__MPI)
  ! Cleanup PAW arrays that are only used for init
  IF (okpaw) CALL paw_post_init() ! only parallel!
#endif
  !
  IF ( lmd ) CALL allocate_dyn_vars()
  !
  ewld = ewald( alat, nat, nsp, ityp, zv, at, bg, tau, &
                omega, g, gg, ngm, gcutm, gstart, gamma_only, strf )
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
  get_nnr = dfftp%nnr
END FUNCTION get_nnr

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

SUBROUTINE set_positions(positions,nat_in)
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
  USE io_global,            ONLY : ionode_id, ionode
  USE mp_images,            ONLY : intra_image_comm
  USE symm_base,            ONLY : checkallsym
  USE mp,                   ONLY : mp_bcast
  USE ener,                 ONLY : ewld

  IMPLICIT NONE
  INTEGER,     INTENT(IN)  :: nat_in
  DOUBLE PRECISION,    INTENT(IN)   :: positions(3,nat_in)

  REAL(DP), EXTERNAL :: ewald

  ! Check consistency of dimensions
  IF (nat_in /= nat) STOP "*** nat provided to set_positions() does not match ions_base%nat"

  !
  !  Update positions on one node
  !
  IF ( ionode ) THEN

     tau = positions

     !
     !  Check that the symmetries have not changed.
     !
     CALL checkallsym( nat, tau, ityp)
  END IF
  !
  !  Sync the positions across all processors
  !
  CALL mp_bcast( tau,       ionode_id, intra_image_comm )

  !
  !  Now, update everything else that depends on the postions
  !

  !
  !  We set treinit_gvecs = .FALSE. to tell hinit1() to keep the same gvecs.
  !
  treinit_gvecs = .FALSE.
  !
  !  Also, set lmovecell = .FALSE. in case someone is using it.
  !
  lmovecell = .FALSE.
  !
  ! ... calculate structure factors for the new positions
  !
  !  NAM: Something like the folling commands would be needed if we want change the
  !  shape and/or size of the cell.  I am not going to try to implement this now.
  !  IF ( lmovecell ) rho%of_g(:,1) = rho%of_g(:,1) / omega
  !  IF ( lmovecell ) CALL scale_h()
  !
  CALL struc_fact( nat, tau, nsp, ityp, ngm, g, bg, &
       dfftp%nr1, dfftp%nr2, dfftp%nr3, strf, eigts1, eigts2, eigts3 )
  !
  !  Update the core charge for the new positions.
  !
  CALL set_rhoc()
  !
  ! ... re-initialize atomic position-dependent quantities
  !
  CALL hinit1()
  !
  !  The ewald energy doesn't seem to be updated by any of the above, so I will
  !  update it here.
  !
  ewld = ewald( alat, nat, nsp, ityp, zv, at, bg, tau, &
                omega, g, gg, ngm, gcutm, gstart, gamma_only, strf )

  RETURN

END SUBROUTINE set_positions


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
  USE scf,           ONLY : rho, rho_core, rhog_core, vltot, v, vrs, kedtau
  USE ener,          ONLY : ehart, etxc, vtxc, epaw, deband
  USE ldaU,          ONLY : eth
  USE fft_rho,       ONLY : rho_r2g, rho_g2r
  USE paw_variables, ONLY : okpaw, ddd_PAW
  USE lsda_mod,      ONLY : nspin
  USE gvecs,         ONLY : doublegrid
  USE paw_onecenter, ONLY : PAW_potential
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
  ! ... plugin contribution to local potential
  !
  CALL plugin_scf_potential(rho,.FALSE.,-1.d0,vltot)

  !
  ! ... compute the potential and store it in v
  !
  CALL v_of_rho( rho, rho_core, rhog_core, &
                 ehart, etxc, vtxc, eth, etotefield, charge, v )
  IF (okpaw) CALL PAW_potential(rho%bec, ddd_PAW, epaw)

  !
  ! ... define the total local potential (external+scf)
  !
  CALL set_vrs( vrs, vltot, v%of_r, kedtau, v%kin_r, dfftp%nnr, nspin, doublegrid )

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

SUBROUTINE v_h_wrapper(rhog, ehart, charge, v,  nnr_in, nspin_in, ngm_in)
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

  IMPLICIT NONE

  INTEGER,     INTENT(IN)  :: ngm_in, nnr_in, nspin_in
  DOUBLE COMPLEX, INTENT(IN)  :: rhog(ngm_in)
  DOUBLE PRECISION,  INTENT(INOUT) :: v(nnr_in,nspin_in)
  DOUBLE PRECISION,    INTENT(OUT) :: ehart, charge

  ! Check consistency of dimensions
  IF (nnr_in /= dfftp%nnr) STOP "*** nnr provided to v_h_wrapper() does not match dfftp%nnr"
  IF (nspin_in /= nspin) STOP "*** nspin provided to v_h_wrapper() does not mach lsda_mod%nspin"
  IF (ngm_in /= ngm) STOP "*** ngm provided to v_h_wrapper() does not match gvect%ngm"

  CALL v_h(rhog, ehart, charge, v)

END SUBROUTINE v_h_wrapper
