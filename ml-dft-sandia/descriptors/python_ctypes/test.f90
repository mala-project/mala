integer(c_int) :: KNOB = 1337

call hello_world()
call create_matrix(2,3)
stop
end

subroutine hello_world()
  write(*,*) "Hello, world!"
  return
end subroutine hello_world

subroutine create_matrix(n1, n2)
  integer, intent(in) :: n1
  integer, intent(in) :: n2
  double precision array(2,3)
  double precision fac

  write(*,*) "Inside create_matrix_"
! uncommenting these lines gives segfault
!   write(*,*) n1,n2
!   fac = 1.0d0/(n1*n2)
!   write(*,*) fac

!   do i = 1, n1
!      do j = 1, n2
!         array(i,j) = fac*(i+1)*(j+1)
!         write(*,*) array(i,j)
!      end do
!   end do
  write(*,*) "Leaving create_matrix_"
  
  return
end subroutine create_matrix
  

