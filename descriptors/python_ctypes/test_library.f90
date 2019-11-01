subroutine hello_world()
  write(*,*) "Hello, world!"
  return
end subroutine hello_world

subroutine create_vector(n, myvec)
  integer, intent(in), value :: n
  double precision, intent(out) :: myvec(n)
  double precision fac

  fac = 100.0d0/n
  
  do i = 1, n
     myvec(i) = fac*i
  end do
  
  return 
end subroutine create_vector

subroutine create_matrix(n1, n2, array)
  integer, intent(in), value :: n1
  integer, intent(in), value :: n2
  double precision, intent(out) :: array(n1, n2)
  double precision fac

  fac = 1.0d0/(n1*n2)
  
  do i = 1, n1
     do j = 1, n2
        array(i,j) = fac*i*j
     end do
  end do
  
  return 
end subroutine create_matrix
