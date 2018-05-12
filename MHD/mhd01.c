#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>

#include <mpi.h>

/*******************************************************************
GSL library  gcc -lgsl -lgslcblas
******************************************************************/
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_const_mksa.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

/*******************************************************************
openMP library  gcc -fopenmp or icc -openmp

mpicc 3d_pic158_domain.c -o 3d_pic158_domain -lm -lgsl -lgslcblas -O3 -openmp -no-prec-div -no-prec-sqrt -xSSE4.2 -static-intel -ipo -opt-mem-bandwidth2 -opt-calloc -unroll-aggressive -ltcmalloc -ansi-alias -opt-subscript-in-range

nohup mpiexec -n 64 -machinefile ../machinefile ./3d_pic112_domain</dev/null&

******************************************************************/
#ifdef _OPENMP
#include <omp.h>
#endif

#define version 1

#define q GSL_CONST_MKSA_ELECTRON_CHARGE
#define mi GSL_CONST_MKSA_MASS_PROTON
#define me GSL_CONST_MKSA_MASS_ELECTRON
#define mu0 GSL_CONST_MKSA_VACUUM_PERMEABILITY
#define e0 GSL_CONST_MKSA_VACUUM_PERMITTIVITY
#define kb GSL_CONST_MKSA_BOLTZMANN
#define Pi M_PI
#define C GSL_CONST_MKSA_SPEED_OF_LIGHT

#define gam 1.5

#define dx 10.
#define dt 10E-9
#define Grid_Nx 64 //grid number x-axis
#define Grid_Ny 64 //grid number y-axis
#define Grid_Nz 64//grid number z-axis

#define V 5E5  //sun wind velocity  z-axis
#define Nall 5E6 //sun wind particle density
#define Ti 1e5  //1eV=1e4
#define Te 1e5

#define R 2.
#define I 1.0e7
#define alpha 0.

#define Step 4000001


typedef struct{
  double bx,by,bz;
  double b0x,b0y,b0z;
  double ex,ey,ez;
  double jx,jy,jz;
  double n;
}Grid;

typedef struct{
  double d[8];
}Mhd;

Mhd U[Grid_Nx+4][Grid_Ny+4][Grid_Nz+4];
Mhd F[Grid_Nx+4][Grid_Ny+4][Grid_Nz+4];
Mhd G[Grid_Nx+4][Grid_Ny+4][Grid_Nz+4];
Mhd H[Grid_Nx+4][Grid_Ny+4][Grid_Nz+4];


int main()
{
  int c;

  Mhd f[Grid_Nx+4][Grid_Ny+4][Grid_Nz+4];
  Mhd g[Grid_Nx+4][Grid_Ny+4][Grid_Nz+4];
  Mhd h[Grid_Nx+4][Grid_Ny+4][Grid_Nz+4];

  for(c=0;c<Step;c++){
    cal_FGH(U);
    
    cat_fgh(f,g,h);
    
    cal_U(f,g,h);

    boundary();
  }
}


int cal_FGH(Mhd u[Grid_Nx+4][Grid_Ny+4][])
{
  int k,l,m;
  double u0,u1,u2,u3,u4,u5,u6;
  double f7_1,f7_2;

  for(k=0;k<Grid_Nx+4;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	u0=1./u[k][l][m].d[0];
	u1=u[k][l][m].d[1]*u[k][l][m].d[1]*u0;
	u2=u[k][l][m].d[2]*u[k][l][m].d[2]*u0;
	u3=u[k][l][m].d[3]*u[k][l][m].d[3]*u0;
	u4=u[k][l][m].d[4]*u[k][l][m].d[4];
	u5=u[k][l][m].d[5]*u[k][l][m].d[5];
	u6=u[k][l][m].d[6]*u[k][l][m].d[6];

	f7_1=(gam*u[k][l][m].d[7]+(1-gam)/2.*(u1+u2+u3)+(2-gam)/2.*(u4+u5+u6))*u0;
	f7_2=(u[k][l][m].d[4]*u[k][l][m].d[1]+u[k][l][m].d[5]*u[k][l][m].d[2]+u[k][l][m].d[6]*u[k][l][m].d[3])*u0;

	F[k][l][m].d[0]=u[k][l][m].d[1];
	G[k][l][m].d[0]=u[k][l][m].d[2];
	H[k][l][m].d[0]=u[k][l][m].d[3];

	F[k][l][m].d[1]=(gam-1.)*u[k][l][m].d[7]+(3.-gam)/2.*u1+(1.-gam)/2.*u2+(1.-gam)/2.*u3;
	G[k][l][m].d[2]=(gam-1.)*u[k][l][m].d[7]+(1.-gam)/2.*u1+(3.-gam)/2.*u2+(1.-gam)/2.*u3;
	H[k][l][m].d[3]=(gam-1.)*u[k][l][m].d[7]+(1.-gam)/2.*u1+(1.-gam)/2.*u2+(3.-gam)/2.*u3;

	F[k][l][m].d[1]+=(-gam)/2.*u4+(2.-gam)/2.*u5+(2.-gam)/2.*u6;
	G[k][l][m].d[2]+=(2.-gam)/2.*u4+(-gam)/2.*u5+(2.-gam)/2.*u6;
	H[k][l][m].d[3]+=(2.-gam)/2.*u4+(2.-gam)/2.*u5+(-gam)/2.*u6;

	F[k][l][m].d[2]=u[k][l][m].d[1]*u[k][l][m].d[2]*u0-u[k][l][m].d[4]*u[k][l][m].d[5];
	G[k][l][m].d[3]=u[k][l][m].d[2]*u[k][l][m].d[3]*u0-u[k][l][m].d[5]*u[k][l][m].d[6];
	H[k][l][m].d[1]=u[k][l][m].d[3]*u[k][l][m].d[1]*u0-u[k][l][m].d[6]*u[k][l][m].d[4];

	F[k][l][m].d[3]=u[k][l][m].d[1]*u[k][l][m].d[3]*u0-u[k][l][m].d[4]*u[k][l][m].d[6];
	G[k][l][m].d[1]=u[k][l][m].d[2]*u[k][l][m].d[1]*u0-u[k][l][m].d[5]*u[k][l][m].d[4];
	H[k][l][m].d[2]=u[k][l][m].d[3]*u[k][l][m].d[2]*u0-u[k][l][m].d[6]*u[k][l][m].d[5];

	F[k][l][m].d[4]=0.;
	G[k][l][m].d[5]=0.;
	H[k][l][m].d[6]=0.;

	F[k][l][m].d[5]=(u[k][l][m].d[5]*u[k][l][m].d[1]-u[k][l][m].d[4]*u[k][l][m].d[2])*u0;
	G[k][l][m].d[6]=(u[k][l][m].d[6]*u[k][l][m].d[2]-u[k][l][m].d[5]*u[k][l][m].d[3])*u0;
	H[k][l][m].d[4]=(u[k][l][m].d[4]*u[k][l][m].d[3]-u[k][l][m].d[6]*u[k][l][m].d[1])*u0;

	F[k][l][m].d[6]=(u[k][l][m].d[6]*u[k][l][m].d[1]-u[k][l][m].d[4]*u[k][l][m].d[3])*u0;
	G[k][l][m].d[4]=(u[k][l][m].d[4]*u[k][l][m].d[2]-u[k][l][m].d[5]*u[k][l][m].d[1])*u0;
	H[k][l][m].d[5]=(u[k][l][m].d[5]*u[k][l][m].d[3]-u[k][l][m].d[6]*u[k][l][m].d[2])*u0;

	F[k][l][m].d[7]=f7_1*u[k][l][m].d[1]-f7_2*u[k][l][m].d[4];
	G[k][l][m].d[7]=f7_1*u[k][l][m].d[2]-f7_2*u[k][l][m].d[5];
	H[k][l][m].d[7]=f7_1*u[k][l][m].d[3]-f7_2*u[k][l][m].d[6];
      }
    }
  }


  return(0);
}

int cal_fgh(Mhd f[Grid_Nx+4][Grid_Ny+4][],Mhd g[Grid_Nx+4][Grid_Ny+4][],Mhd h[Grid_Nx+4][Grid_Ny+4][])
{
  int k,l,m;
  double dxdt=dx/dt;

  for(k=1;k<Grid_Nx+3;k++){
    for(l=1;l<Grid_Ny+3;l++){
      for(m=1;m<Grid_Nz+3;m++){
	f[k][l][m]=0.5*(F[k+1][l][m]+F[k][l][m]-dxdt*(U[k+1][l][m]-U[k][l][m]));
	g[k][l][m]=0.5*(G[k][l+1][m]+G[k][l][m]-dxdt*(U[k][l+1][m]-U[k][l][m]));
	h[k][l][m]=0.5*(H[k][l][m+1]+H[k][l][m]-dxdt*(U[k][l][m+1]-U[k][l][m]));
      }
    }
  }

  return(0);
}


int cal_U(Mhd f[Grid_Nx+4][Grid_Ny+4][],Mhd g[Grid_Nx+4][Grid_Ny+4][],Mhd h[Grid_Nx+4][Grid_Ny+4][])
{
  int k,l,m;
  double dtdx=dt/dx;

  for(k=2;k<Grid_Nx+2;k++){
    for(l=2;l<Grid_Ny+2;l++){
      for(m=2;m<Grid_Nz+2;m++){
	U[k][l][m]+=-dtdx*((f[k][l][m]-f[k-1][l][m])+
			   (g[k][l][m]-g[k][l-1][m])+
			   (h[k][l][m]-h[k][l][m-1]));
      }
    }
  }

  return(0);
}
 
int boundary()
{
  int k,l,m;

  for(k=0;k<2;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	U[k][l][m]=U[Grid_Nx+k][l][m];
	U[Grid_Nx+2+k][l][m]=U[k+2][l][m];
      }
    }
  }

  for(k=2;k<Grid_Nx+2;k++){
    for(l=0;l<2;l++){
      for(m=0;m<Grid_Nz+4;m++){
	U[k][l][m]=U[k][Grid_Ny+l][m];
	U[k][Grid_Ny+2+l][m]=U[k][l+2][m];
      }
    }
  }

  for(k=2;k<Grid_Nx+2;k++){
    for(l=2;l<Grid_Ny+2;l++){
      for(m=0;m<2;m++){
	U[k][l][m]=U[k][l][Grid_Nz+m];
	U[k][l][Grid_Nz+2+m]=U[k][m][m+2];
      }
    }
  }

  return(0);
}
   

