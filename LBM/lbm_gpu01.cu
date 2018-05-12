/*******************************************************************
icc lbm_em27.c -o lbm_em27 -lm -lgsl -lgslcblas -O3 -openmp -no-prec-div -no-prec-sqrt -xSSE4.2 -axSSE4.2 -static-intel -ipo -opt-mem-bandwidth2  -opt-calloc -unroll-aggressive -ltcmalloc

nvcc lbm_gpu01.cu -o lbm_gpu01 -lgsl -lgslcblas -arch sm_20
******************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <time.h>

/*******************************************************************
GSL library  -lgsl -lgslcblas
******************************************************************/
#include <gsl/gsl_const_mksa.h>

#define version 1

#define Nx 100
#define Ny 100
#define Nz 100

#define Dump 10.

#define Tmax 80001
#define Tout 500

#define Pi M_PI

#define C GSL_CONST_MKSA_SPEED_OF_LIGHT //->root 2

#define dx 0.25 //->1
#define dt (dx/(sqrt(2.)*C))  //->1

#define q (GSL_CONST_MKSA_ELECTRON_CHARGE/dt)
#define mi GSL_CONST_MKSA_MASS_PROTON
#define me (GSL_CONST_MKSA_MASS_PROTON/500.)//GSL_CONST_MKSA_MASS_ELECTRON 

#define mu0 (GSL_CONST_MKSA_VACUUM_PERMEABILITY*dt*dt/dx)
#define kb (GSL_CONST_MKSA_BOLTZMANN*dt*dt/(dx*dx))

#define N0 (5.0E6*dx*dx*dx)
#define N1 (6.0E6*dx*dx*dx)
#define V0x (0.*dt/dx)
#define V0y (0.*dt/dx)
#define V0z (0.*dt/dx)

#define Ti 1.0E5
#define Te 1.0E5

#define R (2./dx)
#define I 2e3
#define alpha 0.

#define Inject 2.

#define gamma0 1.
#define nu 0.

//const double omega[3]={2.,2.,2.};
//const double w[7]={1./3.,1./36.,1./36.,1./36.,1./36.,1./18.,1/18.};

/*******************************************************************
type holding cell infomation

i=1,2,3,4,5,6
p=0,1,2 (plane)
j=0,1
s=0,1 (electron,ion)
******************************************************************/
typedef struct{
  //       s  i  p
  double f[2][7][3];
  double f0[2];
  //         s  i  p
  double feq[2][7][3];
  double feq0[2];
  //       s  i  p
  double T[2][7][3];
  double T0[2];
  //       i  p  j
  double g[5][3][2];
  double g0;
  //         i  p  j
  double geq[5][3][2];
  double geq0;

  double rho[2];
  double V[2][3];
  double E[3];
  double B[3];
  double B0[3];
  double J[3];
  double J0[3];

  double rho_c;

  double iota[2];
}Cell;

typedef struct{
  double v[7][3][3];
  double v0[3];
  
  /*
    (0,0,0)
    0 (1,1,0),(-1,1,0),(-1,-1,0),(1,-1,0),(-1,0,0),(1,0,0)
    1 (1,0,1),(-1,0,1),(-1,0,-1),(1,0,-1),(0,-1,0),(0,1,0)
    2 (0,1,1),(0,-1,1),(0,-1,-1),(0,1,-1),(0,0,-1),(0,0,1)
  */
  
  double e[5][3][2][3];
  double e0[3];
  
  /*
    (0,0,0)
    0 0 (1,-1,0),(1,1,0),(-1,1,0),(-1,-1,0)
    0 1 (-1,1,0),(-1,-1,0),(1,-1,0),(1,1,0)
    1 0 (1,0,-1),(1,0,1),(-1,0,1),(-1,0,-1)
    1 1 (-1,0,1),(-1,0,-1),(1,0,-1),(1,0,1)
    2 0 (0,1,-1),(0,1,1),(0,-1,1),(0,-1,-1)
    2 1 (0,-1,1),(0,-1,-1),(0,1,-1),(0,1,1)
  */
  
  double b[5][3][2][3];
  double b0[3];
  
  /*
    (0,0,0)
    0 0 (0,0,-1),(0,0,-1),(0,0,-1),(0,0,-1)
    0 1 (0,0,1),(0,0,1),(0,0,1),(0,0,1)
    1 0 (0,1,0),(0,1,0),(0,1,0),(0,1,0)
    1 1 (0,-1,0),(0,-1,0),(0,-1,0),(0,-1,0)
    2 0 (-1,0,0),(-1,0,0),(-1,0,0),(-1,0,0)
    2 1 (1,0,0),(1,0,0),(1,0,0),(1,0,0)
  */
}Vec;

__global__ void init_dev(Cell *cell,Vec *vec)
{
  int t;
  int k=blockIdx.y;
  int l=threadIdx.x;
  int m=threadIdx.y;

  int offset;
  int offset_kp;
  int offset_km;
  int offset_lp;
  int offset_lm;
  int offset_mp;
  int offset_mm;

  int kp,km,lp,lm,mp,mm;
  int i,p,j,d;
  double x,y,z,r;
  double bx,by,bz;

  double Z[2][3];
  double F[2][3];
  double V[2][3];
  double E[3];
  double J[3];
  double temp[5];

  const double w[7]={1./3.,1./36.,1./36.,1./36.,1./36.,1./18.,1/18.};

  p=0;
  for(i=1;i<5;i++){
    vec->v[i][0][0]=sqrt(2.)*cos(Pi/4.*(2.*i-1.));
    vec->v[i][0][1]=sqrt(2.)*sin(Pi/4.*(2.*i-1.));
    vec->v[i][0][2]=0.;
  }
  for(i=5;i<7;i++){
    vec->v[i][0][0]=pow(-1.,i);
    vec->v[i][0][1]=0.;
    vec->v[i][0][2]=0.;
  }
  
  p=1;
  for(i=1;i<5;i++){
    vec->v[i][1][0]=sqrt(2.)*cos(Pi/4.*(2.*i-1.));
    vec->v[i][1][1]=0.;
    vec->v[i][1][2]=sqrt(2.)*sin(Pi/4.*(2.*i-1.));
  }
  for(i=5;i<7;i++){
    vec->v[i][1][0]=0.;
    vec->v[i][1][1]=pow(-1.,i);
    vec->v[i][1][2]=0.;
  }
  
  p=2;
  for(i=1;i<5;i++){
    vec->v[i][2][0]=0.;
    vec->v[i][2][1]=sqrt(2.)*cos(Pi/4.*(2.*i-1.));
    vec->v[i][2][2]=sqrt(2.)*sin(Pi/4.*(2.*i-1.));
  }
  for(i=5;i<7;i++){
    vec->v[i][2][0]=0.;
    vec->v[i][2][1]=0.;
    vec->v[i][2][2]=pow(-1.,i);
  }
  
  vec->v0[0]=0.;
  vec->v0[1]=0.;
  vec->v0[2]=0.;
  
  for(i=1;i<5;i++){
    for(p=0;p<3;p++){
      j=0;
      vec->e[i][p][0][0]=0.5*vec->v[(i+2)%4+1][p][0];
      vec->e[i][p][0][1]=0.5*vec->v[(i+2)%4+1][p][1];
      vec->e[i][p][0][2]=0.5*vec->v[(i+2)%4+1][p][2];
      
      j=1;
      vec->e[i][p][1][0]=0.5*vec->v[i%4+1][p][0];
      vec->e[i][p][1][1]=0.5*vec->v[i%4+1][p][1];
      vec->e[i][p][1][2]=0.5*vec->v[i%4+1][p][2];
    }
  }
  
  vec->e0[0]=0.;
  vec->e0[1]=0.;
  vec->e0[2]=0.;
  
  for(i=1;i<5;i++){
    for(p=0;p<3;p++){
      for(j=0;j<2;j++){
	vec->b[i][p][j][0]=(vec->v[i][p][1]*vec->e[i][p][j][2]-vec->v[i][p][2]*vec->e[i][p][j][1]);
	vec->b[i][p][j][1]=(vec->v[i][p][2]*vec->e[i][p][j][0]-vec->v[i][p][0]*vec->e[i][p][j][2]);
	vec->b[i][p][j][2]=(vec->v[i][p][0]*vec->e[i][p][j][1]-vec->v[i][p][1]*vec->e[i][p][j][0]);
      }
    }
  }
  
  vec->b0[0]=0.;
  vec->b0[1]=0.;
  vec->b0[2]=0.;

  for(t=0;t<2;t++){
    offset=t*Nx*Ny*Nz+k*Ny*Nz+l*Nz+m;
    
    x=(k-Nx/2-0.5);
    y=(l-Ny/2-0.5);
    z=(m-Nz/2-0.5);
    r=sqrt(x*x+y*y+z*z);
    
    cell[offset].rho[0]=0.;
    cell[offset].rho[1]=0.;
    
    for(d=0;d<3;d++){
      cell[offset].E[d]=0.;
      cell[offset].J[d]=0.;
    }
    
    cell[offset].V[0][0]=V0x;
    cell[offset].V[0][1]=V0y;
    cell[offset].V[0][2]=V0z;
    cell[offset].V[1][0]=V0x;
    cell[offset].V[1][1]=V0y;
    cell[offset].V[1][2]=V0z;
    
    /*
      cell[offset].B0[0]=mu0*I*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5));
      cell[offset].B0[1]=mu0*I*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5));
      cell[offset].B0[2]=mu0*I*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5));
    */
    
    cell[offset].B0[0]=mu0*I/2./Pi*(-(z+R*cos(alpha))
				    /(pow(x+R*sin(alpha),2)+pow(z+R*cos(alpha),2))
				    +(z-R*cos(alpha))
				    /(pow(x-R*sin(alpha),2)+pow(z-R*cos(alpha),2)));
    cell[offset].B0[1]=0.;
    cell[offset].B0[2]=-mu0*I/2./Pi*(-(x+R*sin(alpha))
				     /(pow(x+R*sin(alpha),2)+pow(z+R*cos(alpha),2))
				     +(x-R*sin(alpha))
				     /(pow(x-R*sin(alpha),2)+pow(z-R*cos(alpha),2)));
    
    /*
      cell[offset].B0[0]=0.;
      cell[offset].B0[1]=0.;
      cell[offset.B0[2]=0.;*/
    
    cell[offset].rho_c=0.;
  }

  for(t=0;t<2;t++){
    offset=t*Nx*Ny*Nz+k*Ny*Nz+l*Nz+m;
    kp=(k+1)%Nx; 
    km=(k-1+Nx)%Nx;
    lp=(l+1)%Ny; 
    lm=(l-1+Ny)%Ny;
    mp=(m+1)%Nz; 
    mm=(m-1+Nz)%Nz;
    
    offset_kp=t*Nx*Ny*Nz+kp*Ny*Nz+l*Nz+m;
    offset_km=t*Nx*Ny*Nz+km*Ny*Nz+l*Nz+m;
    offset_lp=t*Nx*Ny*Nz+k*Ny*Nz+lp*Nz+m;
    offset_lm=t*Nx*Ny*Nz+k*Ny*Nz+lm*Nz+m;
    offset_mp=t*Nx*Ny*Nz+k*Ny*Nz+l*Nz+mp;
    offset_mm=t*Nx*Ny*Nz+k*Ny*Nz+l*Nz+mm;
    
    cell[offset].J0[0]=(cell[offset_lp].B0[2]-cell[offset_lm].B0[2])/(2.*mu0)-(cell[offset_mp].B0[1]-cell[offset_mm].B0[1])/(2.*mu0);
    cell[offset].J0[1]=(cell[offset_mp].B0[0]-cell[offset_mm].B0[0])/(2.*mu0)-(cell[offset_kp].B0[2]-cell[offset_km].B0[2])/(2.*mu0);
    cell[offset].J0[2]=(cell[offset_kp].B0[1]-cell[offset_km].B0[1])/(2.*mu0)-(cell[offset_lp].B0[0]-cell[offset_lm].B0[0])/(2.*mu0);
  }

  for(t=0;t<2;t++){
    offset=t*Nx*Ny*Nz+k*Ny*Nz+l*Nz+m;
    
    Z[0][0]=(-q*N0*(+cell[offset].V[0][1]*(cell[offset].B[2]+cell[offset].B0[2])
		    -cell[offset].V[0][2]*(cell[offset].B[1]+cell[offset].B0[1])));
    Z[0][1]=(-q*N0*(+cell[offset].V[0][2]*(cell[offset].B[0]+cell[offset].B0[0])
		    -cell[offset].V[0][0]*(cell[offset].B[2]+cell[offset].B0[2])));
    Z[0][2]=(-q*N0*(+cell[offset].V[0][0]*(cell[offset].B[1]+cell[offset].B0[1])
		    -cell[offset].V[0][1]*(cell[offset].B[0]+cell[offset].B0[0])));
    
    Z[1][0]=(+q*N0*(+cell[offset].V[1][1]*(cell[offset].B[2]+cell[offset].B0[2])
		    -cell[offset].V[1][2]*(cell[offset].B[1]+cell[offset].B0[1])));
    Z[1][1]=(+q*N0*(+cell[offset].V[1][2]*(cell[offset].B[0]+cell[offset].B0[0])
		    -cell[offset].V[1][0]*(cell[offset].B[2]+cell[offset].B0[2])));
    Z[1][2]=(+q*N0*(+cell[offset].V[1][0]*(cell[offset].B[1]+cell[offset].B0[1])
		    -cell[offset].V[1][1]*(cell[offset].B[0]+cell[offset].B0[0])));
    
    temp[0]=1.+mu0*q*q/(8.*mi*mi)*mi*N0;
    temp[1]=  -mu0*q*q/(8.*me*mi)*me*N0;
    temp[2]=1.+mu0*q*q/(8.*me*me)*me*N0;
    temp[3]=  -mu0*q*q/(8.*me*mi)*mi*N0;
    temp[4]=1.+0.125*mu0*q*q*(me*N0/(me*me)+mi*N0/(mi*mi));
    
    F[0][0]=(temp[0]*Z[0][0]-temp[1]*Z[1][0])/temp[4];
    F[0][1]=(temp[0]*Z[0][1]-temp[1]*Z[1][1])/temp[4];
    F[0][2]=(temp[0]*Z[0][2]-temp[1]*Z[1][2])/temp[4];
    
    F[1][0]=(temp[2]*Z[1][0]-temp[3]*Z[0][0])/temp[4];
    F[1][1]=(temp[2]*Z[1][1]-temp[3]*Z[0][1])/temp[4];
    F[1][2]=(temp[2]*Z[1][2]-temp[3]*Z[0][2])/temp[4];
    
    V[0][0]=cell[offset].V[0][0]+F[0][0]/(2.*me*N0);
    V[0][1]=cell[offset].V[0][1]+F[0][1]/(2.*me*N0);
    V[0][2]=cell[offset].V[0][2]+F[0][2]/(2.*me*N0);
    
    V[1][0]=cell[offset].V[1][0]+F[1][0]/(2.*mi*N0);
    V[1][1]=cell[offset].V[1][1]+F[1][1]/(2.*mi*N0);
    V[1][2]=cell[offset].V[1][2]+F[1][2]/(2.*mi*N0);
    
    cell[offset].iota[0]=N0*kb*Te*pow(me*N0,-gamma0);
    cell[offset].iota[1]=N0*kb*Ti*pow(mi*N0,-gamma0);
    
    for(i=1;i<7;i++){
      for(p=0;p<3;p++){
	cell[offset].f[0][i][p]=w[i]*me*N0*(3.*cell[offset].iota[0]*pow(me*N0,gamma0-1.)
					    +3.*(vec->v[i][p][0]*V[0][0]+vec->v[i][p][1]*V[0][1]+vec->v[i][p][2]*V[0][2])
					    +4.5*pow((vec->v[i][p][0]*V[0][0]+vec->v[i][p][1]*V[0][1]+vec->v[i][p][2]*V[0][2]),2)
					    -1.5*(V[0][0]*V[0][0]+V[0][1]*V[0][1]+V[0][2]*V[0][2]));
	cell[offset].f[1][i][p]=w[i]*mi*N0*(3.*cell[offset].iota[1]*pow(mi*N0,gamma0-1.)
					    +3.*(vec->v[i][p][0]*V[1][0]+vec->v[i][p][1]*V[1][1]+vec->v[i][p][2]*V[1][2])
					    +4.5*pow((vec->v[i][p][0]*V[1][0]+vec->v[i][p][1]*V[1][1]+vec->v[i][p][2]*V[1][2]),2)
					    -1.5*(V[1][0]*V[1][0]+V[1][1]*V[1][1]+V[1][2]*V[1][2]));
	
	cell[offset].feq[0][i][p]=0.;
	cell[offset].feq[1][i][p]=0.;
	
	cell[offset].T[0][i][p]=0.;
	cell[offset].T[1][i][p]=0.;
      }
    }
    
    cell[offset].f0[0]=3.*w[0]*me*N0*(1.-0.5*(4.*cell[offset].iota[0]*pow(me*N0,gamma0-1.)
					      +(V[0][0]*V[0][0]+V[0][1]*V[0][1]+V[0][2]*V[0][2])));
    cell[offset].f0[1]=3.*w[0]*mi*N0*(1.-0.5*(4.*cell[offset].iota[1]*pow(mi*N0,gamma0-1.)
					      +(V[1][0]*V[1][0]+V[1][1]*V[1][1]+V[1][2]*V[1][2])));
    
    cell[offset].feq0[0]=0.;
    cell[offset].feq0[1]=0.;
    
    cell[offset].T0[0]=0.;
    cell[offset].T0[1]=0.;
    
    J[0]=0.5*(-q*F[0][0]/me+q*F[1][0]/mi);
    J[1]=0.5*(-q*F[0][1]/me+q*F[1][1]/mi);
    J[2]=0.5*(-q*F[0][2]/me+q*F[1][2]/mi);
    
    E[0]=-0.25*mu0*J[0];
    E[1]=-0.25*mu0*J[1];
    E[2]=-0.25*mu0*J[2];
    
    for(i=1;i<5;i++){
      for(p=0;p<3;p++){
	for(j=0;j<2;j++){
	  bx=by=bz=0.;
	  
	  cell[offset].g[i][p][j]=(E[0]*vec->e[i][p][j][0]+E[1]*vec->e[i][p][j][1]+E[2]*vec->e[i][p][j][2])/4.+(bx*vec->b[i][p][j][0]+by*vec->b[i][p][j][1]+bz*vec->b[i][p][j][2])/8.;
	  cell[offset].geq[i][p][j]=0.;
	}
      }
    }
    
    cell[offset].g0=0.;
    cell[offset].geq0=0.;
    
    if(m>=80&&m<90&&k>90&&k<=110){
      Z[0][0]=(-q*N1*(+cell[offset].V[0][1]*(cell[offset].B[2]+cell[offset].B0[2])
		      -cell[offset].V[0][2]*(cell[offset].B[1]+cell[offset].B0[1])));
      Z[0][1]=(-q*N1*(+cell[offset].V[0][2]*(cell[offset].B[0]+cell[offset].B0[0])
		      -cell[offset].V[0][0]*(cell[offset].B[2]+cell[offset].B0[2])));
      Z[0][2]=(-q*N1*(+cell[offset].V[0][0]*(cell[offset].B[1]+cell[offset].B0[1])
		      -cell[offset].V[0][1]*(cell[offset].B[0]+cell[offset].B0[0])));
      
      Z[1][0]=(+q*N1*(+cell[offset].V[1][1]*(cell[offset].B[2]+cell[offset].B0[2])
		      -cell[offset].V[1][2]*(cell[offset].B[1]+cell[offset].B0[1])));
      Z[1][1]=(+q*N1*(+cell[offset].V[1][2]*(cell[offset].B[0]+cell[offset].B0[0])
		      -cell[offset].V[1][0]*(cell[offset].B[2]+cell[offset].B0[2])));
      Z[1][2]=(+q*N1*(+cell[offset].V[1][0]*(cell[offset].B[1]+cell[offset].B0[1])
		      -cell[offset].V[1][1]*(cell[offset].B[0]+cell[offset].B0[0])));
      
      temp[0]=1.+mu0*q*q/(8.*mi*mi)*mi*N1;
      temp[1]=  -mu0*q*q/(8.*me*mi)*me*N1;
      temp[2]=1.+mu0*q*q/(8.*me*me)*me*N1;
      temp[3]=  -mu0*q*q/(8.*me*mi)*mi*N1;
      temp[4]=1.+0.125*mu0*q*q*(me*N1/(me*me)+mi*N1/(mi*mi));
      
      F[0][0]=(temp[0]*Z[0][0]-temp[1]*Z[1][0])/temp[4];
      F[0][1]=(temp[0]*Z[0][1]-temp[1]*Z[1][1])/temp[4];
      F[0][2]=(temp[0]*Z[0][2]-temp[1]*Z[1][2])/temp[4];
      
      F[1][0]=(temp[2]*Z[1][0]-temp[3]*Z[0][0])/temp[4];
      F[1][1]=(temp[2]*Z[1][1]-temp[3]*Z[0][1])/temp[4];
      F[1][2]=(temp[2]*Z[1][2]-temp[3]*Z[0][2])/temp[4];
      
      V[0][0]=cell[offset].V[0][0]+F[0][0]/(2.*me*N1);
      V[0][1]=cell[offset].V[0][1]+F[0][1]/(2.*me*N1);
      V[0][2]=cell[offset].V[0][2]+F[0][2]/(2.*me*N1);
      
      V[1][0]=cell[offset].V[1][0]+F[1][0]/(2.*mi*N1);
      V[1][1]=cell[offset].V[1][1]+F[1][1]/(2.*mi*N1);
      V[1][2]=cell[offset].V[1][2]+F[1][2]/(2.*mi*N1);
      
      cell[offset].iota[0]=N1*kb*Te*pow(me*N1,-gamma0);
      cell[offset].iota[1]=N1*kb*Ti*pow(mi*N1,-gamma0);
      
      for(i=1;i<7;i++){
	for(p=0;p<3;p++){
	  cell[offset].f[0][i][p]=w[i]*me*N1*(3.*cell[offset].iota[0]*pow(me*N1,gamma0-1.)
					      +3.*(vec->v[i][p][0]*V[0][0]+vec->v[i][p][1]*V[0][1]+vec->v[i][p][2]*V[0][2])
					      +4.5*pow((vec->v[i][p][0]*V[0][0]+vec->v[i][p][1]*V[0][1]+vec->v[i][p][2]*V[0][2]),2)
					      -1.5*(V[0][0]*V[0][0]+V[0][1]*V[0][1]+V[0][2]*V[0][2]));
	  cell[offset].f[1][i][p]=w[i]*mi*N1*(3.*cell[offset].iota[1]*pow(mi*N1,gamma0-1.)
					      +3.*(vec->v[i][p][0]*V[1][0]+vec->v[i][p][1]*V[1][1]+vec->v[i][p][2]*V[1][2])
					      +4.5*pow((vec->v[i][p][0]*V[1][0]+vec->v[i][p][1]*V[1][1]+vec->v[i][p][2]*V[1][2]),2)
					      -1.5*(V[1][0]*V[1][0]+V[1][1]*V[1][1]+V[1][2]*V[1][2]));
	  
	  cell[offset].feq[0][i][p]=0.;
	  cell[offset].feq[1][i][p]=0.;
	  
	  cell[offset].T[0][i][p]=0.;
	  cell[offset].T[1][i][p]=0.;
	}
      }
      
      cell[offset].f0[0]=3.*w[0]*me*N1*(1.-0.5*(4.*cell[offset].iota[0]*pow(me*N1,gamma0-1.)
						+(V[0][0]*V[0][0]+V[0][1]*V[0][1]+V[0][2]*V[0][2])));
      cell[offset].f0[1]=3.*w[0]*mi*N1*(1.-0.5*(4.*cell[offset].iota[1]*pow(mi*N1,gamma0-1.)
						+(V[1][0]*V[1][0]+V[1][1]*V[1][1]+V[1][2]*V[1][2])));
      
      cell[offset].feq0[0]=0.;
      cell[offset].feq0[1]=0.;
      
      cell[offset].T0[0]=0.;
      cell[offset].T0[1]=0.;
      
      J[0]=0.5*(-q*F[0][0]/me+q*F[1][0]/mi);
      J[1]=0.5*(-q*F[0][1]/me+q*F[1][1]/mi);
      J[2]=0.5*(-q*F[0][2]/me+q*F[1][2]/mi);
      
      E[0]=-0.25*mu0*J[0];
      E[1]=-0.25*mu0*J[1];
      E[2]=-0.25*mu0*J[2];
      
      for(i=1;i<5;i++){
	for(p=0;p<3;p++){
	  for(j=0;j<2;j++){
	    bx=by=bz=0.;
	    
	    cell[offset].g[i][p][j]=(E[0]*vec->e[i][p][j][0]+E[1]*vec->e[i][p][j][1]+E[2]*vec->e[i][p][j][2])/4.+(bx*vec->b[i][p][j][0]+by*vec->b[i][p][j][1]+bz*vec->b[i][p][j][2])/8.;
	    cell[offset].geq[i][p][j]=0.;
	  }
	}
      }
      
      cell[offset].g0=0.;
      cell[offset].geq0=0.;
    }else if(m>111&&m<=121&&k>90&&k<=110){
      Z[0][0]=(-q*N1*(+cell[offset].V[0][1]*(cell[offset].B[2]+cell[offset].B0[2])
		      -cell[offset].V[0][2]*(cell[offset].B[1]+cell[offset].B0[1])));
      Z[0][1]=(-q*N1*(+cell[offset].V[0][2]*(cell[offset].B[0]+cell[offset].B0[0])
		      -cell[offset].V[0][0]*(cell[offset].B[2]+cell[offset].B0[2])));
      Z[0][2]=(-q*N1*(+cell[offset].V[0][0]*(cell[offset].B[1]+cell[offset].B0[1])
		      -cell[offset].V[0][1]*(cell[offset].B[0]+cell[offset].B0[0])));
      
      Z[1][0]=(+q*N1*(+cell[offset].V[1][1]*(cell[offset].B[2]+cell[offset].B0[2])
		      -cell[offset].V[1][2]*(cell[offset].B[1]+cell[offset].B0[1])));
      Z[1][1]=(+q*N1*(+cell[offset].V[1][2]*(cell[offset].B[0]+cell[offset].B0[0])
		      -cell[offset].V[1][0]*(cell[offset].B[2]+cell[offset].B0[2])));
      Z[1][2]=(+q*N1*(+cell[offset].V[1][0]*(cell[offset].B[1]+cell[offset].B0[1])
		      -cell[offset].V[1][1]*(cell[offset].B[0]+cell[offset].B0[0])));
      
      temp[0]=1.+mu0*q*q/(8.*mi*mi)*mi*N1;
      temp[1]=  -mu0*q*q/(8.*me*mi)*me*N1;
      temp[2]=1.+mu0*q*q/(8.*me*me)*me*N1;
      temp[3]=  -mu0*q*q/(8.*me*mi)*mi*N1;
      temp[4]=1.+0.125*mu0*q*q*(me*N1/(me*me)+mi*N1/(mi*mi));
      
      F[0][0]=(temp[0]*Z[0][0]-temp[1]*Z[1][0])/temp[4];
      F[0][1]=(temp[0]*Z[0][1]-temp[1]*Z[1][1])/temp[4];
      F[0][2]=(temp[0]*Z[0][2]-temp[1]*Z[1][2])/temp[4];
      
      F[1][0]=(temp[2]*Z[1][0]-temp[3]*Z[0][0])/temp[4];
      F[1][1]=(temp[2]*Z[1][1]-temp[3]*Z[0][1])/temp[4];
      F[1][2]=(temp[2]*Z[1][2]-temp[3]*Z[0][2])/temp[4];
      
      V[0][0]=cell[offset].V[0][0]+F[0][0]/(2.*me*N1);
      V[0][1]=cell[offset].V[0][1]+F[0][1]/(2.*me*N1);
      V[0][2]=cell[offset].V[0][2]+F[0][2]/(2.*me*N1);
      
      V[1][0]=cell[offset].V[1][0]+F[1][0]/(2.*mi*N1);
      V[1][1]=cell[offset].V[1][1]+F[1][1]/(2.*mi*N1);
      V[1][2]=cell[offset].V[1][2]+F[1][2]/(2.*mi*N1);
      
      cell[offset].iota[0]=N1*kb*Te*pow(me*N1,-gamma0);
      cell[offset].iota[1]=N1*kb*Ti*pow(mi*N1,-gamma0);
      
      for(i=1;i<7;i++){
	for(p=0;p<3;p++){
	  cell[offset].f[0][i][p]=w[i]*me*N1*(3.*cell[offset].iota[0]*pow(me*N1,gamma0-1.)
					      +3.*(vec->v[i][p][0]*V[0][0]+vec->v[i][p][1]*V[0][1]+vec->v[i][p][2]*V[0][2])
					      +4.5*pow((vec->v[i][p][0]*V[0][0]+vec->v[i][p][1]*V[0][1]+vec->v[i][p][2]*V[0][2]),2)
					      -1.5*(V[0][0]*V[0][0]+V[0][1]*V[0][1]+V[0][2]*V[0][2]));
	  cell[offset].f[1][i][p]=w[i]*mi*N1*(3.*cell[offset].iota[1]*pow(mi*N1,gamma0-1.)
					      +3.*(vec->v[i][p][0]*V[1][0]+vec->v[i][p][1]*V[1][1]+vec->v[i][p][2]*V[1][2])
					      +4.5*pow((vec->v[i][p][0]*V[1][0]+vec->v[i][p][1]*V[1][1]+vec->v[i][p][2]*V[1][2]),2)
					      -1.5*(V[1][0]*V[1][0]+V[1][1]*V[1][1]+V[1][2]*V[1][2]));
	  
	  cell[offset].feq[0][i][p]=0.;
	  cell[offset].feq[1][i][p]=0.;
	  
	  cell[offset].T[0][i][p]=0.;
	  cell[offset].T[1][i][p]=0.;
	}
      }
      
      cell[offset].f0[0]=3.*w[0]*me*N1*(1.-0.5*(4.*cell[offset].iota[0]*pow(me*N1,gamma0-1.)
						+(V[0][0]*V[0][0]+V[0][1]*V[0][1]+V[0][2]*V[0][2])));
      cell[offset].f0[1]=3.*w[0]*mi*N1*(1.-0.5*(4.*cell[offset].iota[1]*pow(mi*N1,gamma0-1.)
						+(V[1][0]*V[1][0]+V[1][1]*V[1][1]+V[1][2]*V[1][2])));
      
      cell[offset].feq0[0]=0.;
      cell[offset].feq0[1]=0.;
      
      cell[offset].T0[0]=0.;
      cell[offset].T0[1]=0.;
      
      J[0]=0.5*(-q*F[0][0]/me+q*F[1][0]/mi);
      J[1]=0.5*(-q*F[0][1]/me+q*F[1][1]/mi);
      J[2]=0.5*(-q*F[0][2]/me+q*F[1][2]/mi);
      
      E[0]=-0.25*mu0*J[0];
      E[1]=-0.25*mu0*J[1];
      E[2]=-0.25*mu0*J[2];
      
      for(i=1;i<5;i++){
	for(p=0;p<3;p++){
	  for(j=0;j<2;j++){
	    bx=by=bz=0.;
	    
	    cell[offset].g[i][p][j]=(E[0]*vec->e[i][p][j][0]+E[1]*vec->e[i][p][j][1]+E[2]*vec->e[i][p][j][2])/4.+(bx*vec->b[i][p][j][0]+by*vec->b[i][p][j][1]+bz*vec->b[i][p][j][2])/8.;
	    cell[offset].geq[i][p][j]=0.;
	  }
	}
      }
      
      cell[offset].g0=0.;
      cell[offset].geq0=0.;
    }
  }
}

__global__ void cal_val(Cell *cell,Vec *vec,int tpp)
{
  int k=blockIdx.y;
  int l=threadIdx.x;
  int m=threadIdx.y;
  int i,p,j,s,d;

  int offset_p=tpp*Nx*Ny*Nz+k*Ny*Nz+l*Nz+m;
  
  cell[offset_p].rho[0]=cell[offset_p].f0[0];
  cell[offset_p].rho[1]=cell[offset_p].f0[1];
  cell[offset_p].rho_c=0.;
  
  for(d=0;d<3;d++){
    cell[offset_p].V[0][d]=0.;
    cell[offset_p].V[1][d]=0.;
    cell[offset_p].E[d]=0.;
    cell[offset_p].B[d]=0.;
    cell[offset_p].J[d]=0.;
  }
  
  for(s=0;s<2;s++){
    for(i=1;i<7;i++){
      for(p=0;p<3;p++){
	cell[offset_p].rho[s]+=cell[offset_p].f[s][i][p];
	
	for(d=0;d<3;d++){
	  cell[offset_p].V[s][d]+=cell[offset_p].f[s][i][p]*vec->v[i][p][d];
	}
      }
    }
  }
  
  for(s=0;s<2;s++){
    for(d=0;d<3;d++){
      cell[offset_p].V[s][d]=cell[offset_p].V[s][d]/cell[offset_p].rho[s];
    }
  }
  
  for(i=1;i<5;i++){
    for(p=0;p<3;p++){
      for(j=0;j<2;j++){
	for(d=0;d<3;d++){
	  cell[offset_p].E[d]+=cell[offset_p].g[i][p][j]*vec->e[i][p][j][d];
	  cell[offset_p].B[d]+=cell[offset_p].g[i][p][j]*vec->b[i][p][j][d];
	}
      }
    }
  }
  
  for(d=0;d<3;d++){
    cell[offset_p].J[d]=-q/me*cell[offset_p].rho[0]*cell[offset_p].V[0][d]+q/mi*cell[offset_p].rho[1]*cell[offset_p].V[1][d];
  }
  
  cell[offset_p].rho_c=-q/me*cell[offset_p].rho[0]+q/mi*cell[offset_p].rho[1];
}

__global__ void cal_T(Cell *cell,Vec *vec,int tpp)
{
  int k=blockIdx.y;
  int l=threadIdx.x;
  int m=threadIdx.y;
  int i,p,s,j;
  double Z[2][3];
  double F[2][3];
  double V[2][3];
  double E[3];
  double J[3];
  double temp[5];

  int offset_p=tpp*Nx*Ny*Nz+k*Ny*Nz+l*Nz+m;

  const double omega[3]={2.,2.,2.};
  const double w[7]={1./3.,1./36.,1./36.,1./36.,1./36.,1./18.,1/18.};
  
  Z[0][0]=(-q/me*cell[offset_p].rho[0]*(cell[offset_p].E[0]-0.25*mu0*cell[offset_p].J[0]
					+cell[offset_p].V[0][1]*(cell[offset_p].B[2]+cell[offset_p].B0[2])
					-cell[offset_p].V[0][2]*(cell[offset_p].B[1]+cell[offset_p].B0[1]))
	   -nu*cell[offset_p].rho[0]*(cell[offset_p].V[0][0]-cell[offset_p].V[1][0]));
  Z[0][1]=(-q/me*cell[offset_p].rho[0]*(cell[offset_p].E[1]-0.25*mu0*cell[offset_p].J[1]
					+cell[offset_p].V[0][2]*(cell[offset_p].B[0]+cell[offset_p].B0[0])
					-cell[offset_p].V[0][0]*(cell[offset_p].B[2]+cell[offset_p].B0[2]))
	   -nu*cell[offset_p].rho[0]*(cell[offset_p].V[0][1]-cell[offset_p].V[1][1]));
  Z[0][2]=(-q/me*cell[offset_p].rho[0]*(cell[offset_p].E[2]-0.25*mu0*cell[offset_p].J[2]
					+cell[offset_p].V[0][0]*(cell[offset_p].B[1]+cell[offset_p].B0[1])
					-cell[offset_p].V[0][1]*(cell[offset_p].B[0]+cell[offset_p].B0[0]))
	   -nu*cell[offset_p].rho[0]*(cell[offset_p].V[0][2]-cell[offset_p].V[1][2]));
  
  Z[1][0]=(+q/mi*cell[offset_p].rho[1]*(cell[offset_p].E[0]-0.25*mu0*cell[offset_p].J[0]
					+cell[offset_p].V[1][1]*(cell[offset_p].B[2]+cell[offset_p].B0[2])
					-cell[offset_p].V[1][2]*(cell[offset_p].B[1]+cell[offset_p].B0[1]))
	   -nu*cell[offset_p].rho[1]*(cell[offset_p].V[1][0]-cell[offset_p].V[0][0]));
  Z[1][1]=(+q/mi*cell[offset_p].rho[1]*(cell[offset_p].E[1]-0.25*mu0*cell[offset_p].J[1]
					+cell[offset_p].V[1][2]*(cell[offset_p].B[0]+cell[offset_p].B0[0])
					-cell[offset_p].V[1][0]*(cell[offset_p].B[2]+cell[offset_p].B0[2]))
	   -nu*cell[offset_p].rho[1]*(cell[offset_p].V[1][1]-cell[offset_p].V[0][1]));
  Z[1][2]=(+q/mi*cell[offset_p].rho[1]*(cell[offset_p].E[2]-0.25*mu0*cell[offset_p].J[2]
					+cell[offset_p].V[1][0]*(cell[offset_p].B[1]+cell[offset_p].B0[1])
					-cell[offset_p].V[1][1]*(cell[offset_p].B[0]+cell[offset_p].B0[0]))
	   -nu*cell[offset_p].rho[1]*(cell[offset_p].V[1][2]-cell[offset_p].V[0][2]));
  
  temp[0]=1.+mu0*q*q/(8.*mi*mi)*cell[offset_p].rho[1];
  temp[1]=  -mu0*q*q/(8.*me*mi)*cell[offset_p].rho[0];
  temp[2]=1.+mu0*q*q/(8.*me*me)*cell[offset_p].rho[0];
  temp[3]=  -mu0*q*q/(8.*me*mi)*cell[offset_p].rho[1];
  temp[4]=1.+0.125*mu0*q*q*(cell[offset_p].rho[0]/(me*me)+cell[offset_p].rho[1]/(mi*mi));
  
  F[0][0]=(temp[0]*Z[0][0]-temp[1]*Z[1][0])/temp[4];
  F[0][1]=(temp[0]*Z[0][1]-temp[1]*Z[1][1])/temp[4];
  F[0][2]=(temp[0]*Z[0][2]-temp[1]*Z[1][2])/temp[4];
  
  F[1][0]=(temp[2]*Z[1][0]-temp[3]*Z[0][0])/temp[4];
  F[1][1]=(temp[2]*Z[1][1]-temp[3]*Z[0][1])/temp[4];
  F[1][2]=(temp[2]*Z[1][2]-temp[3]*Z[0][2])/temp[4];
  
  V[0][0]=cell[offset_p].V[0][0]+F[0][0]/(2.*cell[offset_p].rho[0]);
  V[0][1]=cell[offset_p].V[0][1]+F[0][1]/(2.*cell[offset_p].rho[0]);
  V[0][2]=cell[offset_p].V[0][2]+F[0][2]/(2.*cell[offset_p].rho[0]);
  
  V[1][0]=cell[offset_p].V[1][0]+F[1][0]/(2.*cell[offset_p].rho[1]);
  V[1][1]=cell[offset_p].V[1][1]+F[1][1]/(2.*cell[offset_p].rho[1]);
  V[1][2]=cell[offset_p].V[1][2]+F[1][2]/(2.*cell[offset_p].rho[1]);
  
  for(s=0;s<2;s++){
    for(i=1;i<7;i++){
      for(p=0;p<3;p++){
	cell[offset_p].T[s][i][p]=(1.-omega[s]/2.)*w[i]*(3.*((vec->v[i][p][0]-V[s][0])*F[s][0]+
							     (vec->v[i][p][1]-V[s][1])*F[s][1]+
							     (vec->v[i][p][2]-V[s][2])*F[s][2])+
							 9.*(((vec->v[i][p][0]*V[s][0])+
							      (vec->v[i][p][1]*V[s][1])+
							      (vec->v[i][p][2]*V[s][2]))*
							     ((vec->v[i][p][0]*F[s][0])+
							      (vec->v[i][p][1]*F[s][1])+
							      (vec->v[i][p][2]*F[s][2]))));
      }
    }
  }
  
  cell[offset_p].T0[0]=(1.-omega[0]/2.)*w[0]*(-3.*(V[0][0]*F[0][0]+
						   V[0][1]*F[0][1]+
						   V[0][2]*F[0][2]));
  cell[offset_p].T0[1]=(1.-omega[1]/2.)*w[0]*(-3.*(V[1][0]*F[1][0]+
						   V[1][1]*F[1][1]+
						   V[1][2]*F[1][2]));
  
  for(s=0;s<2;s++){
    for(i=1;i<7;i++){
      for(p=0;p<3;p++){
	cell[offset_p].feq[s][i][p]=w[i]*cell[offset_p].rho[s]
	  *(3.*cell[offset_p].iota[s]*pow(cell[offset_p].rho[s],gamma0-1.)
	    +3.*((vec->v[i][p][0]*V[s][0])+(vec->v[i][p][1]*V[s][1])+(vec->v[i][p][2]*V[s][2]))
	    +4.5*pow((vec->v[i][p][0]*V[s][0])+(vec->v[i][p][1]*V[s][1])+(vec->v[i][p][2]*V[s][2]),2.)
	    -1.5*(V[s][0]*V[s][0]+V[s][1]*V[s][1]+V[s][2]*V[s][2]));
      }
    }
  }
  
  cell[offset_p].feq0[0]=3.*w[0]*cell[offset_p].rho[0]
    *(1.-0.5*(4.*cell[offset_p].iota[0]*pow(cell[offset_p].rho[0],gamma0-1.)
	      +(V[0][0]*V[0][0]+V[0][1]*V[0][1]+V[0][2]*V[0][2])));
  
  cell[offset_p].feq0[1]=3.*w[0]*cell[offset_p].rho[1]
    *(1.-0.5*(4.*cell[offset_p].iota[1]*pow(cell[offset_p].rho[1],gamma0-1.)
	      +(V[1][0]*V[1][0]+V[1][1]*V[1][1]+V[1][2]*V[1][2])));
  
  J[0]=cell[offset_p].J[0]+0.5*(-q*F[0][0]/me+q*F[1][0]/mi);
  J[1]=cell[offset_p].J[1]+0.5*(-q*F[0][1]/me+q*F[1][1]/mi);
  J[2]=cell[offset_p].J[2]+0.5*(-q*F[0][2]/me+q*F[1][2]/mi);
  
  E[0]=cell[offset_p].E[0]-0.25*mu0*J[0];
  E[1]=cell[offset_p].E[1]-0.25*mu0*J[1];
  E[2]=cell[offset_p].E[2]-0.25*mu0*J[2];
  
  for(i=1;i<5;i++){
    for(p=0;p<3;p++){
      for(j=0;j<2;j++){
	cell[offset_p].geq[i][p][j]=0.25*(E[0]*vec->e[i][p][j][0]+
					  E[1]*vec->e[i][p][j][1]+
					  E[2]*vec->e[i][p][j][2])
	  +0.125*(cell[offset_p].B[0]*vec->b[i][p][j][0]+
		  cell[offset_p].B[1]*vec->b[i][p][j][1]+
		  cell[offset_p].B[2]*vec->b[i][p][j][2]);
      }
    }
  }
  
  cell[offset_p].geq0=0.;
}

__global__ void step(Cell *cell,int tpp,int tnx)
{
  int k=blockIdx.y;
  int l=threadIdx.x;
  int m=threadIdx.y;

  int s;

  int offset_p=tpp*Nx*Ny*Nz+k*Ny*Nz+l*Nz+m;

  int kp=(k+1)%Nx; 
  int km=(k-1+Nx)%Nx;
  int lp=(l+1)%Ny; 
  int lm=(l-1+Ny)%Ny;
  int mp=(m+1)%Nz; 
  int mm=(m-1+Nz)%Nz;

  int offset_n=tnx*Nx*Ny*Nz+k*Ny*Nz+l*Nz+m;

  int offset_kp=tnx*Nx*Ny*Nz+kp*Ny*Nz+l*Nz+m;
  int offset_km=tnx*Nx*Ny*Nz+km*Ny*Nz+l*Nz+m;
  int offset_lp=tnx*Nx*Ny*Nz+k*Ny*Nz+lp*Nz+m;
  int offset_lm=tnx*Nx*Ny*Nz+k*Ny*Nz+lm*Nz+m;
  int offset_mp=tnx*Nx*Ny*Nz+k*Ny*Nz+l*Nz+mp;
  int offset_mm=tnx*Nx*Ny*Nz+k*Ny*Nz+l*Nz+mm;

  int offset_kp_lp=tnx*Nx*Ny*Nz+kp*Ny*Nz+lp*Nz+m;
  int offset_km_lp=tnx*Nx*Ny*Nz+km*Ny*Nz+lp*Nz+m;
  int offset_km_lm=tnx*Nx*Ny*Nz+km*Ny*Nz+lm*Nz+m;
  int offset_kp_lm=tnx*Nx*Ny*Nz+kp*Ny*Nz+lm*Nz+m;

  int offset_kp_mp=tnx*Nx*Ny*Nz+kp*Ny*Nz+l*Nz+mp;
  int offset_km_mp=tnx*Nx*Ny*Nz+km*Ny*Nz+l*Nz+mp;
  int offset_km_mm=tnx*Nx*Ny*Nz+km*Ny*Nz+l*Nz+mm;
  int offset_kp_mm=tnx*Nx*Ny*Nz+kp*Ny*Nz+l*Nz+mm;

  int offset_lp_mp=tnx*Nx*Ny*Nz+k*Ny*Nz+lp*Nz+mp;
  int offset_lm_mp=tnx*Nx*Ny*Nz+k*Ny*Nz+lm*Nz+mp;
  int offset_lm_mm=tnx*Nx*Ny*Nz+k*Ny*Nz+lm*Nz+mm;
  int offset_lp_mm=tnx*Nx*Ny*Nz+k*Ny*Nz+lp*Nz+mm;

  const double omega[3]={2.,2.,2.};
	
  for(s=0;s<2;s++){
    cell[offset_n].f0[s]=cell[offset_p].f0[s]-omega[s]*(cell[offset_p].f0[s]-cell[offset_p].feq0[s])+cell[offset_p].T0[s];
    
    cell[offset_kp_lp].f[s][1][0]=cell[offset_p].f[s][1][0]-omega[s]*(cell[offset_p].f[s][1][0]-cell[offset_p].feq[s][1][0])+cell[offset_p].T[s][1][0];
    cell[offset_km_lp].f[s][2][0]=cell[offset_p].f[s][2][0]-omega[s]*(cell[offset_p].f[s][2][0]-cell[offset_p].feq[s][2][0])+cell[offset_p].T[s][2][0];
    cell[offset_km_lm].f[s][3][0]=cell[offset_p].f[s][3][0]-omega[s]*(cell[offset_p].f[s][3][0]-cell[offset_p].feq[s][3][0])+cell[offset_p].T[s][3][0];
    cell[offset_kp_lm].f[s][4][0]=cell[offset_p].f[s][4][0]-omega[s]*(cell[offset_p].f[s][4][0]-cell[offset_p].feq[s][4][0])+cell[offset_p].T[s][4][0];
    cell[offset_km   ].f[s][5][0]=cell[offset_p].f[s][5][0]-omega[s]*(cell[offset_p].f[s][5][0]-cell[offset_p].feq[s][5][0])+cell[offset_p].T[s][5][0];
    cell[offset_kp   ].f[s][6][0]=cell[offset_p].f[s][6][0]-omega[s]*(cell[offset_p].f[s][6][0]-cell[offset_p].feq[s][6][0])+cell[offset_p].T[s][6][0];
    
    cell[offset_kp_mp].f[s][1][1]=cell[offset_p].f[s][1][1]-omega[s]*(cell[offset_p].f[s][1][1]-cell[offset_p].feq[s][1][1])+cell[offset_p].T[s][1][1];
    cell[offset_km_mp].f[s][2][1]=cell[offset_p].f[s][2][1]-omega[s]*(cell[offset_p].f[s][2][1]-cell[offset_p].feq[s][2][1])+cell[offset_p].T[s][2][1];
    cell[offset_km_mm].f[s][3][1]=cell[offset_p].f[s][3][1]-omega[s]*(cell[offset_p].f[s][3][1]-cell[offset_p].feq[s][3][1])+cell[offset_p].T[s][3][1];
    cell[offset_kp_mm].f[s][4][1]=cell[offset_p].f[s][4][1]-omega[s]*(cell[offset_p].f[s][4][1]-cell[offset_p].feq[s][4][1])+cell[offset_p].T[s][4][1];
    cell[offset_lm   ].f[s][5][1]=cell[offset_p].f[s][5][1]-omega[s]*(cell[offset_p].f[s][5][1]-cell[offset_p].feq[s][5][1])+cell[offset_p].T[s][5][1];
    cell[offset_lp   ].f[s][6][1]=cell[offset_p].f[s][6][1]-omega[s]*(cell[offset_p].f[s][6][1]-cell[offset_p].feq[s][6][1])+cell[offset_p].T[s][6][1];
    
    cell[offset_lp_mp].f[s][1][2]=cell[offset_p].f[s][1][2]-omega[s]*(cell[offset_p].f[s][1][2]-cell[offset_p].feq[s][1][2])+cell[offset_p].T[s][1][2];
    cell[offset_lm_mp].f[s][2][2]=cell[offset_p].f[s][2][2]-omega[s]*(cell[offset_p].f[s][2][2]-cell[offset_p].feq[s][2][2])+cell[offset_p].T[s][2][2];
    cell[offset_lm_mm].f[s][3][2]=cell[offset_p].f[s][3][2]-omega[s]*(cell[offset_p].f[s][3][2]-cell[offset_p].feq[s][3][2])+cell[offset_p].T[s][3][2];
    cell[offset_lp_mm].f[s][4][2]=cell[offset_p].f[s][4][2]-omega[s]*(cell[offset_p].f[s][4][2]-cell[offset_p].feq[s][4][2])+cell[offset_p].T[s][4][2];
    cell[offset_mm   ].f[s][5][2]=cell[offset_p].f[s][5][2]-omega[s]*(cell[offset_p].f[s][5][2]-cell[offset_p].feq[s][5][2])+cell[offset_p].T[s][5][2];
    cell[offset_mp   ].f[s][6][2]=cell[offset_p].f[s][6][2]-omega[s]*(cell[offset_p].f[s][6][2]-cell[offset_p].feq[s][6][2])+cell[offset_p].T[s][6][2];
  }
  
  cell[offset_n].g0=cell[offset_p].g0-omega[2]*(cell[offset_p].g0-cell[offset_p].geq0);
  
  cell[offset_kp_lp].g[1][0][0]=cell[offset_p].g[1][0][0]-omega[2]*(cell[offset_p].g[1][0][0]-cell[offset_p].geq[1][0][0]);
  cell[offset_km_lp].g[2][0][0]=cell[offset_p].g[2][0][0]-omega[2]*(cell[offset_p].g[2][0][0]-cell[offset_p].geq[2][0][0]);
  cell[offset_km_lm].g[3][0][0]=cell[offset_p].g[3][0][0]-omega[2]*(cell[offset_p].g[3][0][0]-cell[offset_p].geq[3][0][0]);
  cell[offset_kp_lm].g[4][0][0]=cell[offset_p].g[4][0][0]-omega[2]*(cell[offset_p].g[4][0][0]-cell[offset_p].geq[4][0][0]);
  
  cell[offset_kp_lp].g[1][0][1]=cell[offset_p].g[1][0][1]-omega[2]*(cell[offset_p].g[1][0][1]-cell[offset_p].geq[1][0][1]);
  cell[offset_km_lp].g[2][0][1]=cell[offset_p].g[2][0][1]-omega[2]*(cell[offset_p].g[2][0][1]-cell[offset_p].geq[2][0][1]);
  cell[offset_km_lm].g[3][0][1]=cell[offset_p].g[3][0][1]-omega[2]*(cell[offset_p].g[3][0][1]-cell[offset_p].geq[3][0][1]);
  cell[offset_kp_lm].g[4][0][1]=cell[offset_p].g[4][0][1]-omega[2]*(cell[offset_p].g[4][0][1]-cell[offset_p].geq[4][0][1]);
  
  cell[offset_kp_mp].g[1][1][0]=cell[offset_p].g[1][1][0]-omega[2]*(cell[offset_p].g[1][1][0]-cell[offset_p].geq[1][1][0]);
  cell[offset_km_mp].g[2][1][0]=cell[offset_p].g[2][1][0]-omega[2]*(cell[offset_p].g[2][1][0]-cell[offset_p].geq[2][1][0]);
  cell[offset_km_mm].g[3][1][0]=cell[offset_p].g[3][1][0]-omega[2]*(cell[offset_p].g[3][1][0]-cell[offset_p].geq[3][1][0]);
  cell[offset_kp_mm].g[4][1][0]=cell[offset_p].g[4][1][0]-omega[2]*(cell[offset_p].g[4][1][0]-cell[offset_p].geq[4][1][0]);
  
  cell[offset_kp_mp].g[1][1][1]=cell[offset_p].g[1][1][1]-omega[2]*(cell[offset_p].g[1][1][1]-cell[offset_p].geq[1][1][1]);
  cell[offset_km_mp].g[2][1][1]=cell[offset_p].g[2][1][1]-omega[2]*(cell[offset_p].g[2][1][1]-cell[offset_p].geq[2][1][1]);
  cell[offset_km_mm].g[3][1][1]=cell[offset_p].g[3][1][1]-omega[2]*(cell[offset_p].g[3][1][1]-cell[offset_p].geq[3][1][1]);
  cell[offset_kp_mm].g[4][1][1]=cell[offset_p].g[4][1][1]-omega[2]*(cell[offset_p].g[4][1][1]-cell[offset_p].geq[4][1][1]);
  
  cell[offset_lp_mp].g[1][2][0]=cell[offset_p].g[1][2][0]-omega[2]*(cell[offset_p].g[1][2][0]-cell[offset_p].geq[1][2][0]);
  cell[offset_lm_mp].g[2][2][0]=cell[offset_p].g[2][2][0]-omega[2]*(cell[offset_p].g[2][2][0]-cell[offset_p].geq[2][2][0]);
  cell[offset_lm_mm].g[3][2][0]=cell[offset_p].g[3][2][0]-omega[2]*(cell[offset_p].g[3][2][0]-cell[offset_p].geq[3][2][0]);
  cell[offset_lp_mm].g[4][2][0]=cell[offset_p].g[4][2][0]-omega[2]*(cell[offset_p].g[4][2][0]-cell[offset_p].geq[4][2][0]);
  
  cell[offset_lp_mp].g[1][2][1]=cell[offset_p].g[1][2][1]-omega[2]*(cell[offset_p].g[1][2][1]-cell[offset_p].geq[1][2][1]);
  cell[offset_lm_mp].g[2][2][1]=cell[offset_p].g[2][2][1]-omega[2]*(cell[offset_p].g[2][2][1]-cell[offset_p].geq[2][2][1]);
  cell[offset_lm_mm].g[3][2][1]=cell[offset_p].g[3][2][1]-omega[2]*(cell[offset_p].g[3][2][1]-cell[offset_p].geq[3][2][1]);
  cell[offset_lp_mm].g[4][2][1]=cell[offset_p].g[4][2][1]-omega[2]*(cell[offset_p].g[4][2][1]-cell[offset_p].geq[4][2][1]);
}

__global__ void dump(Cell *cell,Vec *vec,int t)
{
  int k=blockIdx.y;
  int l=threadIdx.x;
  int m=threadIdx.y;
  int i,p,j;
  double d;

  double Z[2][3];
  double F[2][3];
  double V[2][3];
  double temp[5];

  int offset=t*Nx*Ny*Nz+k*Ny*Nz+l*Nz+m;

  const double w[7]={1./3.,1./36.,1./36.,1./36.,1./36.,1./18.,1/18.};
  
  if(m==0){
    cell[offset].V[0][0]=V0x;
    cell[offset].V[0][1]=V0y;
    cell[offset].V[0][2]=V0z;
    cell[offset].V[1][0]=V0x;
    cell[offset].V[1][1]=V0y;
    cell[offset].V[1][2]=V0z;
    
    Z[0][0]=(-q*N0*(cell[offset].V[0][1]*(cell[offset].B[2]+cell[offset].B0[2])
		    -cell[offset].V[0][2]*(cell[offset].B[1]+cell[offset].B0[1])));
    Z[0][1]=(-q*N0*(cell[offset].V[0][2]*(cell[offset].B[0]+cell[offset].B0[0])
		    -cell[offset].V[0][0]*(cell[offset].B[2]+cell[offset].B0[2])));
    Z[0][2]=(-q*N0*(cell[offset].V[0][0]*(cell[offset].B[1]+cell[offset].B0[1])
		    -cell[offset].V[0][1]*(cell[offset].B[0]+cell[offset].B0[0])));
    
    Z[1][0]=(+q*N0*(cell[offset].V[1][1]*(cell[offset].B[2]+cell[offset].B0[2])
		    -cell[offset].V[1][2]*(cell[offset].B[1]+cell[offset].B0[1])));
    Z[1][1]=(+q*N0*(cell[offset].V[1][2]*(cell[offset].B[0]+cell[offset].B0[0])
		    -cell[offset].V[1][0]*(cell[offset].B[2]+cell[offset].B0[2])));
    Z[1][2]=(+q*N0*(cell[offset].V[1][0]*(cell[offset].B[1]+cell[offset].B0[1])
		    -cell[offset].V[1][1]*(cell[offset].B[0]+cell[offset].B0[0])));
    
    temp[0]=1.+mu0*q*q/(8.*mi*mi)*mi*N0;
    temp[1]=  -mu0*q*q/(8.*me*mi)*me*N0;
    temp[2]=1.+mu0*q*q/(8.*me*me)*me*N0;
    temp[3]=  -mu0*q*q/(8.*me*mi)*mi*N0;
    temp[4]=1.+0.125*mu0*q*q*(me*N0/(me*me)+mi*N0/(mi*mi));
    
    F[0][0]=(temp[0]*Z[0][0]-temp[1]*Z[1][0])/temp[4];
    F[0][1]=(temp[0]*Z[0][1]-temp[1]*Z[1][1])/temp[4];
    F[0][2]=(temp[0]*Z[0][2]-temp[1]*Z[1][2])/temp[4];
    
    F[1][0]=(temp[2]*Z[1][0]-temp[3]*Z[0][0])/temp[4];
    F[1][1]=(temp[2]*Z[1][1]-temp[3]*Z[0][1])/temp[4];
    F[1][2]=(temp[2]*Z[1][2]-temp[3]*Z[0][2])/temp[4];
    
    V[0][0]=cell[offset].V[0][0]+F[0][0]/(2.*me*N0);
    V[0][1]=cell[offset].V[0][1]+F[0][1]/(2.*me*N0);
    V[0][2]=cell[offset].V[0][2]+F[0][2]/(2.*me*N0);
    
    V[1][0]=cell[offset].V[1][0]+F[1][0]/(2.*mi*N0);
    V[1][1]=cell[offset].V[1][1]+F[1][1]/(2.*mi*N0);
    V[1][2]=cell[offset].V[1][2]+F[1][2]/(2.*mi*N0);
    
    cell[offset].iota[0]=N0*kb*Te*pow(me*N0,-gamma0);
    cell[offset].iota[1]=N0*kb*Ti*pow(mi*N0,-gamma0);
    
    for(i=1;i<7;i++){
      for(p=0;p<3;p++){
	cell[offset].f[0][i][p]=w[i]*me*N0*(3.*cell[offset].iota[0]*pow(me*N0,gamma0-1.)
					    +3.*(vec->v[i][p][0]*V[0][0]+vec->v[i][p][1]*V[0][1]+vec->v[i][p][2]*V[0][2])
					    +4.5*pow((vec->v[i][p][0]*V[0][0]+vec->v[i][p][1]*V[0][1]+vec->v[i][p][2]*V[0][2]),2)
					    -1.5*(V[0][0]*V[0][0]+V[0][1]*V[0][1]+V[0][2]*V[0][2]));
	cell[offset].f[1][i][p]=w[i]*mi*N0*(3.*cell[offset].iota[1]*pow(mi*N0,gamma0-1.)
					    +3.*(vec->v[i][p][0]*V[1][0]+vec->v[i][p][1]*V[1][1]+vec->v[i][p][2]*V[1][2])
					    +4.5*pow((vec->v[i][p][0]*V[1][0]+vec->v[i][p][1]*V[1][1]+vec->v[i][p][2]*V[1][2]),2)
					    -1.5*(V[1][0]*V[1][0]+V[1][1]*V[1][1]+V[1][2]*V[1][2]));
      }
    }
    
    cell[offset].f0[0]=3.*w[0]*me*N0*(1.-0.5*(4.*cell[offset].iota[0]*pow(me*N0,gamma0-1.)
					      +(V[0][0]*V[0][0]+V[0][1]*V[0][1]+V[0][2]*V[0][2])));
    cell[offset].f0[1]=3.*w[0]*mi*N0*(1.-0.5*(4.*cell[offset].iota[1]*pow(mi*N0,gamma0-1.)
					      +(V[1][0]*V[1][0]+V[1][1]*V[1][1]+V[1][2]*V[1][2])));
  }else if(m==Nz-1){
    for(i=1;i<7;i++){
      for(p=0;p<3;p++){
	cell[offset].f[0][i][p]=cell[offset-1].f[0][i][p];
	cell[offset].f[1][i][p]=cell[offset-1].f[1][i][p];
      }
    }
    
    cell[offset].f0[0]=cell[offset-1].f0[0];
    cell[offset].f0[1]=cell[offset-1].f0[1];
  }
  
  d=1.;
  
  /*if(k<Dump){
    d=d*pow((double)k/Dump,4.);
    }else if(k>Nx-1-Dump){
    d=d*pow((double)(Nx-1-k)/Dump,4.);
    }*/
  
  if(m<Dump){
    d=d*pow((double)m/Dump,4.);
  }else if(m>Nz-1-Dump){
    d=d*pow((double)(Nz-1-m)/Dump,4.);
  }
  
  if(d<1.){
    //fprintf(stderr,"%f\n",d);
    cell[offset].E[0]=d*cell[offset].E[0];
    cell[offset].E[1]=d*cell[offset].E[1];
    cell[offset].E[2]=d*cell[offset].E[2];
    cell[offset].B[0]=d*cell[offset].B[0];
    cell[offset].B[1]=d*cell[offset].B[1];
    cell[offset].B[2]=d*cell[offset].B[2];
    
    for(i=1;i<5;i++){
      for(p=0;p<3;p++){
	for(j=0;j<2;j++){
	  cell[offset].g[i][p][j]=((cell[offset].E[0]*vec->e[i][p][j][0]+cell[offset].E[1]*vec->e[i][p][j][1]+cell[offset].E[2]*vec->e[i][p][j][2])/4.+
				   (cell[offset].B[0]*vec->b[i][p][j][0]+cell[offset].B[1]*vec->b[i][p][j][1]+cell[offset].B[2]*vec->b[i][p][j][2])/8.);
	  
	}
      }
    }
  }//
}

int init_host(Cell *cell,Cell *dev_cell,Vec *dev_vec)
{
  int t,k,l,m;
  int offset;
  const double omega[3]={2.,2.,2.};

  fprintf(stderr,"viscous     : %E\n",(1/omega[0]-0.5)/3.);
  fprintf(stderr,"L           : %E\n",pow(mu0*pow(Pi*R*R*I,2)/(8*Pi*Pi*N0*mi*V0z*V0z),1./6.)*dx);
  fprintf(stderr,"B@mp        : %E\n",sqrt(2*mu0*N0*mi*V0z*V0z)/(dt*dt));
  fprintf(stderr,"v_th        : %E %E\n",sqrt(2.*kb*Ti/mi)*dx/dt,sqrt(2.*kb*Te/me)*dx/dt);
  fprintf(stderr,"vs          : %E\n",sqrt(2.*kb*Ti/mi+kb*Te/mi)*dx/dt);
  fprintf(stderr,"mach number : %E\n",V0z/sqrt(2.*kb*Ti/mi+kb*Te/mi));

  cell=(Cell*)malloc(2*Nx*Ny*Nz*sizeof(Cell));
  cudaMalloc((void**)&dev_cell,2*Nx*Ny*Nz*sizeof(Cell));
  cudaMalloc((void**)&dev_vec,1*sizeof(Vec));

  if(cell==NULL){
    printf("err\n");
  }

  fprintf(stderr,"%d\n",sizeof(Cell));

  for(t=0;t<2;t++){
    for(k=0;k<Nx;k++){
      for(l=0;l<Ny;l++){
	for(m=0;m<Nz;m++){
	  offset=t*Nx*Ny*Nz+k*Ny*Nz+l*Nz+m;
	  cell[offset].B[0]=0.;
	}
      }
    }
  }

  return(0);
}

int time_evolution(Cell *dev_cell,Vec *dev_vec,int *tpp,int *tnx)
{
  dim3 blocks(1,Nx);
  dim3 threads(Ny,Nz);

  cal_val<<<blocks,threads>>>(dev_cell,dev_vec,*tpp);

  cal_T<<<blocks,threads>>>(dev_cell,dev_vec,*tpp);

  step<<<blocks,threads>>>(dev_cell,*tpp,*tnx);

  return(0);
}

int output(Cell *cell,int c,int tnx)
{
  int k,l,m;
  FILE *fp1;
  char filename[256];
  int offset_n;
  
  sprintf(filename,"lbm%03d-%d.dat",version,c);
  fp1=fopen(filename,"w");
  
  fprintf(fp1,"VARIABLES = \"X[m]\" \"Y[m]\" \"Z[m]\" \"Ne[/m3]\" \"Ni[/m3]\" \"Pe[N/m2]\" \"Pi[N/m2]\" \"Vx_e[m/s]\" \"Vy_e[m/s]\"  \"Vz_e[m/s]\" \"Vx_i[m/s]\"  \"Vy_i[m/s]\" \"Vz_i[m/s]\" \"rhoVx_e[]\" \"rhoVy_e[m/s]\"  \"rhoVz_e[m/s]\" \"rhoVx_i[m/s]\"  \"rhoVy_i[m/s]\" \"rhoVz_i[m/s]\"\"Ex[V/m]\" \"Ey[V/m]\" \"Ez[V/m]\" \"Bx[T]\" \"By[T]\" \"Bz[T]\" \"Bpx[T]\" \"Bpy[T]\" \"Bpz[T]\" \"B[T]\" \"Jx[A/m2]\" \"Jy[A/m2]\" \"Jz[A/m2]\" \n");
  fprintf(fp1,"ZONE T=\"STP:%d\", STRANDID=1, SOLUTIONTIME=%d, I=%d, J=%d, K=%d\n",c,c,Nz,Ny,Nx);
  
  for(k=0;k<Nx;k++){
    for(l=0;l<Ny;l++){
      for(m=0;m<Nz;m++){
	offset_n=tnx*Nx*Ny*Nz+k*Ny*Nz+l*Nz+m;

	fprintf(fp1,"%.3E %.3E %.3E %.8E %.8E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E  %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E\n",
		k*dx,l*dx,m*dx,
		cell[offset_n].rho[0]/me/(dx*dx*dx),
		cell[offset_n].rho[1]/mi/(dx*dx*dx),
		cell[offset_n].iota[0]*pow(cell[offset_n].rho[0],gamma0)/(dx*dt*dt),
		cell[offset_n].iota[0]*pow(cell[offset_n].rho[0],gamma0)/(dx*dt*dt),
		cell[offset_n].V[0][0]*dx/dt,
		cell[offset_n].V[0][1]*dx/dt,
		cell[offset_n].V[0][2]*dx/dt,
		cell[offset_n].V[1][0]*dx/dt,
		cell[offset_n].V[1][1]*dx/dt,
		cell[offset_n].V[1][2]*dx/dt,
		cell[offset_n].rho[0]/(dx*dx*dx)*cell[offset_n].V[0][0]*dx/dt,
		cell[offset_n].rho[0]/(dx*dx*dx)*cell[offset_n].V[0][1]*dx/dt,
		cell[offset_n].rho[0]/(dx*dx*dx)*cell[offset_n].V[0][2]*dx/dt,
		cell[offset_n].rho[1]/(dx*dx*dx)*cell[offset_n].V[1][0]*dx/dt,
		cell[offset_n].rho[1]/(dx*dx*dx)*cell[offset_n].V[1][1]*dx/dt,
		cell[offset_n].rho[1]/(dx*dx*dx)*cell[offset_n].V[1][2]*dx/dt,
		cell[offset_n].E[0]*dx/(dt*dt*dt),
		cell[offset_n].E[1]*dx/(dt*dt*dt),
		cell[offset_n].E[2]*dx/(dt*dt*dt),
		(cell[offset_n].B0[0]+cell[offset_n].B[0])/(dt*dt),
		(cell[offset_n].B0[1]+cell[offset_n].B[1])/(dt*dt),
		(cell[offset_n].B0[2]+cell[offset_n].B[2])/(dt*dt),
		cell[offset_n].B[0]/(dt*dt),
		cell[offset_n].B[1]/(dt*dt),
		cell[offset_n].B[2]/(dt*dt),
		sqrt(pow(cell[offset_n].B0[0]+cell[offset_n].B[0],2)+
		     pow(cell[offset_n].B0[1]+cell[offset_n].B[1],2)+
		     pow(cell[offset_n].B0[2]+cell[offset_n].B[2],2))/(dt*dt),
		cell[offset_n].J[0]/(dx*dx),
		cell[offset_n].J[1]/(dx*dx),
		cell[offset_n].J[2]/(dx*dx));
	
      }
      fprintf(fp1,"\n");
    }
  }
  
  fclose(fp1);

  return(0);
}

int output_gnuplot(Cell *cell,int c,int tnx)
{
  int k,l,m;
  FILE *fp1;
  char filename[256];
  int offset_n;
  
  sprintf(filename,"lbm%03d-%d_gnuplot.dat",version,c);
  fp1=fopen(filename,"w");
  
  l=0;
  for(k=0;k<Nx;k++){
    for(m=0;m<Nz;m++){
      offset_n=tnx*Nx*Ny*Nz+k*Ny*Nz+l*Nz+m;

      fprintf(fp1,"%.3E %.3E %.3E %.8E %.8E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E\n",
	      k*dx,l*dx,m*dx,//123
	      cell[offset_n].rho[0]/me/(dx*dx*dx),//4
	      cell[offset_n].rho[1]/mi/(dx*dx*dx),//5
	      cell[offset_n].iota[0]*pow(cell[offset_n].rho[0],gamma0)/(dx*dt*dt),//6
	      cell[offset_n].iota[0]*pow(cell[offset_n].rho[0],gamma0)/(dx*dt*dt),//7
	      cell[offset_n].V[0][0]*dx/dt,//8
	      cell[offset_n].V[0][1]*dx/dt,//9
	      cell[offset_n].V[0][2]*dx/dt,//10
	      cell[offset_n].V[1][0]*dx/dt,//11
	      cell[offset_n].V[1][1]*dx/dt,//12
	      cell[offset_n].V[1][2]*dx/dt,//13
	      cell[offset_n].rho[0]/(dx*dx*dx)*cell[offset_n].V[0][0]*dx/dt,//14
	      cell[offset_n].rho[0]/(dx*dx*dx)*cell[offset_n].V[0][1]*dx/dt,//15
	      cell[offset_n].rho[0]/(dx*dx*dx)*cell[offset_n].V[0][2]*dx/dt,//16
	      cell[offset_n].rho[1]/(dx*dx*dx)*cell[offset_n].V[1][0]*dx/dt,//17
	      cell[offset_n].rho[1]/(dx*dx*dx)*cell[offset_n].V[1][1]*dx/dt,//18
	      cell[offset_n].rho[1]/(dx*dx*dx)*cell[offset_n].V[1][2]*dx/dt,//19
	      cell[offset_n].E[0]*dx/(dt*dt*dt),//20
	      cell[offset_n].E[1]*dx/(dt*dt*dt),//21
	      cell[offset_n].E[2]*dx/(dt*dt*dt),//22
	      (cell[offset_n].B0[0]+cell[offset_n].B[0])/(dt*dt),//23
	      (cell[offset_n].B0[1]+cell[offset_n].B[1])/(dt*dt),//24
	      (cell[offset_n].B0[2]+cell[offset_n].B[2])/(dt*dt),//25
	      cell[offset_n].B[0]/(dt*dt),//26
	      cell[offset_n].B[1]/(dt*dt),//27
	      cell[offset_n].B[2]/(dt*dt),//28
	      sqrt(pow(cell[offset_n].B0[0]+cell[offset_n].B[0],2)+
		   pow(cell[offset_n].B0[1]+cell[offset_n].B[1],2)+
		   pow(cell[offset_n].B0[2]+cell[offset_n].B[2],2))/(dt*dt),//29
	      cell[offset_n].J[0]/(dx*dx),//30
	      cell[offset_n].J[1]/(dx*dx),//31
	      cell[offset_n].J[2]/(dx*dx),//32
	      cell[offset_n].J0[0]/(dx*dx),//33
	      cell[offset_n].J0[1]/(dx*dx),//34
	      cell[offset_n].J0[2]/(dx*dx));//35
      
    }
    fprintf(fp1,"\n");
  }
  
  fclose(fp1);

  return(0);
}

int main()
{
  int t;

  int tpp=0;
  int tnx=1;
  int offset_n;

  Cell *cell;
  Cell *dev_cell;
  Vec *dev_vec;

  dim3 blocks(1,Nx);
  dim3 threads(Ny,Nz);

  cell=NULL;
  dev_cell=NULL;
  dev_vec=NULL;

  init_host(cell,dev_cell,dev_vec);
  init_dev<<<blocks,threads>>>(dev_cell,dev_vec);

  cal_val<<<blocks,threads>>>(dev_cell,dev_vec,tpp);

  //cudaMemcpy(cell,dev_cell,2*Nx*Ny*Nz*sizeof(Cell),cudaMemcpyDeviceToHost); 
  //output_gnuplot(cell,-1,tnx);

  for(t=0;t<Tmax;t++){
    time_evolution(dev_cell,dev_vec,&tpp,&tnx);

    tpp=tnx;
    tnx=(tnx+1)%2;

    if(t%Tout==0){
      cudaMemcpy(cell,dev_cell,2*Nx*Ny*Nz*sizeof(Cell),cudaMemcpyDeviceToHost);

      offset_n=tnx*Nx*Ny*Nz+Nx/2*Ny*Nz+Ny/2*Nz+Nz/2;
      fprintf(stderr,"%d\n",t);
      //fprintf(stderr,"%f %f %E\n",cell[offset_n].feq[1][1][0]/cell[offset_n].f[1][1][0],cell[offset_n].geq[1][1][0]/cell[offset_n].g[1][1][0],cell[offset_n].rho[0]/me/(dx*dx*dx));
 
      //output(cell,t,tnx);
    }
  }

  return(0);
}
