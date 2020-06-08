
SUBROUTINE total_energy_setup()
  ! set up modules used by v_xc and v_h
  USE gvect,       ONLY : ngm
  USE lsda_mod,    ONLY : nspin
  USE kinds,       ONLY : DP
  USE scf,         ONLY : scf_type
  USE fft_base,    ONLY : dfftp
  IMPLICIT NONE
  print *, "Setup called"
  nspin = 2
END SUBROUTINE
!
INTEGER FUNCTION get_nnr()
  ! Get the value of dfftp%nnr, which is an argument to x_xc_wrapper() 
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


SUBROUTINE v_xc_wrapper(rho_of_r, rho_of_g, rho_core, rhog_core,&
                         etxc, vtxc, v, nnr_in, nspin_in, ngm_in)
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
  REAL(DP),    INTENT(IN)    :: rho_of_r(nnr_in,nspin_in), rho_core(nnr_in)
  COMPLEX(DP), INTENT(IN)    :: rho_of_g(ngm_in,nspin_in), rhog_core(ngm_in)
  REAL(DP),    INTENT(OUT)   :: v(nnr_in,nspin_in), vtxc, etxc

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
  COMPLEX(DP), INTENT(IN)  :: rhog(ngm_in)
  REAL(DP),  INTENT(INOUT) :: v(nnr_in,nspin_in)
  REAL(DP),    INTENT(OUT) :: ehart, charge

  ! Check consistency of dimensions
  IF (nnr_in /= dfftp%nnr) STOP "*** nnr provided to v_h_wrapper() does not match dfftp%nnr"
  IF (nspin_in /= nspin) STOP "*** nspin provided to v_h_wrapper() does not mach lsda_mod%nspin"
  IF (ngm_in /= ngm) STOP "*** ngm provided to v_h_wrapper() does not match gvect%ngm"

  CALL v_h(rhog, ehart, charge, v)

END SUBROUTINE v_h_wrapper

