/*******************************************************************
icc lbm_em21.c -o lbm_em21 -lm -lgsl -lgslcblas -O3 -openmp -no-prec-div -no-prec-sqrt -xSSE4.2 -axSSE4.2 -static-intel -ipo -opt-mem-bandwidth2  -opt-calloc -unroll-aggressive -ltcmalloc
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
#include <gsl/gsl_math.h>
#include <gsl/gsl_const_mksa.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define version 21

#define Nx 100
#define Ny 1
#define Nz 100

#define Dump 10.

#define Tmax 2000001

#define Pi M_PI

#define C GSL_CONST_MKSA_SPEED_OF_LIGHT //->root 2

#define dx 50000. //->1
#define dt (dx/(sqrt(2.)*C))  //->1

#define q (GSL_CONST_MKSA_ELECTRON_CHARGE/dt)
#define mi GSL_CONST_MKSA_MASS_PROTON
#define me (GSL_CONST_MKSA_MASS_PROTON/100.)//GSL_CONST_MKSA_MASS_ELECTRON 

#define mu0 (GSL_CONST_MKSA_VACUUM_PERMEABILITY*dt*dt/dx)
#define kb (GSL_CONST_MKSA_BOLTZMANN*dt*dt/(dx*dx))

#define N0 (5.0E6*dx*dx*dx)
#define V0x (0.*dt/dx)
#define V0y (0.*dt/dx)
#define V0z (5.E5*dt/dx)

#define Ti 1.0E5
#define Te 1.0E5

#define R (250500./dx)
#define I 0.//2e3
#define alpha 0.

#define Inject 2.

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
  double rho_c;

  double iota[2];
}Cell;

typedef struct{
  int k,l,m;
}klm;

Cell ****cell;

double omega[3];
double gamma0;
double nu;

double w[7];

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

Vec vec;

int tpp=0;
int tnx=1;

double integrate_bx(int k,int l,int m);
double integrate_by(int k,int l,int m);
double integrate_bz(int k,int l,int m);
double fbx(double w,void *params);
double fby(double w,void *params);
double fbz(double w,void *params);

int main()
{
  int t;

  init();

  cal_val();

  output(-1);

  for(t=0;t<Tmax;t++){
    time_evolution(t);

    if(t%100==0){
      fprintf(stderr,"%d\n",t);
      fprintf(stderr,"%f %f %E\n",cell[tnx][Nx/2][Ny/2][Nz/2].feq[1][1][0]/cell[tnx][Nx/2][Ny/2][Nz/2].f[1][1][0],cell[tnx][Nx/2][Ny/2][Nz/2].geq[1][1][0]/cell[tnx][Nx/2][Ny/2][Nz/2].g[1][1][0],cell[tnx][Nx/2][Ny/2][Nz/2].rho[0]/me/(dx*dx*dx));
      //fprintf(stderr,"%f %f %E\n",cell[tnx][Nx/2][Ny/2][Nz/2].feq[0][3][0]/cell[tnx][Nx/2][Ny/2][Nz/2].f[0][3][0],cell[tnx][Nx/2][Ny/2][Nz/2].geq[1][1][0]/cell[tnx][Nx/2][Ny/2][Nz/2].g[1][1][0],cell[tnx][Nx/2][Ny/2][Nz/2].rho[0]/me);
      output(t);
    }
  }

  return(0);
}

int init()
{
  int t,k,l,m;
  int i,p,j,s,d;
  double x,y,z,r;
  double bx,by,bz;

  double Z[2][3];
  double F[2][3];
  double V[2][3];
  double E[3];
  double J[3];
  double temp[5];

  omega[0]=1.5;//0.5;
  omega[1]=1.5;//0.5;
  omega[2]=2.;
  gamma0=1.;
  nu=0.;

  fprintf(stderr,"viscous     : %E\n",(1/omega[0]-0.5)/3.);
  fprintf(stderr,"L           : %E\n",pow(mu0*pow(Pi*R*R*I,2)/(8*Pi*Pi*N0*mi*V0z*V0z),1./6.)*dx);
  fprintf(stderr,"B@mp        : %E\n",sqrt(2*mu0*N0*mi*V0z*V0z)/(dt*dt));
  fprintf(stderr,"v_th        : %E %E\n",sqrt(2.*kb*Ti/mi)*dx/dt,sqrt(2.*kb*Te/me)*dx/dt);
  fprintf(stderr,"vs          : %E\n",sqrt(2.*kb*Ti/mi+kb*Te/mi)*dx/dt);
  fprintf(stderr,"mach number : %E\n",V0z/sqrt(2.*kb*Ti/mi+kb*Te/mi));

  w[0]=1./3.;
  w[1]=1./36.;
  w[2]=1./36.;
  w[3]=1./36.;
  w[4]=1./36.;
  w[5]=1./18.;
  w[6]=1./18.;

  p=0;
  for(i=1;i<5;i++){
    vec.v[i][0][0]=sqrt(2.)*cos(Pi/4.*(2.*i-1.));
    vec.v[i][0][1]=sqrt(2.)*sin(Pi/4.*(2.*i-1.));
    vec.v[i][0][2]=0.;
  }
  for(i=5;i<7;i++){
    vec.v[i][0][0]=pow(-1,i);
    vec.v[i][0][1]=0.;
    vec.v[i][0][2]=0.;
  }
  
  p=1;
  for(i=1;i<5;i++){
    vec.v[i][1][0]=sqrt(2.)*cos(Pi/4.*(2.*i-1.));
    vec.v[i][1][1]=0.;
    vec.v[i][1][2]=sqrt(2.)*sin(Pi/4.*(2.*i-1.));
  }
  for(i=5;i<7;i++){
    vec.v[i][1][0]=0.;
    vec.v[i][1][1]=pow(-1,i);
    vec.v[i][1][2]=0.;
  }
  
  p=2;
  for(i=1;i<5;i++){
    vec.v[i][2][0]=0.;
    vec.v[i][2][1]=sqrt(2.)*cos(Pi/4.*(2.*i-1.));
    vec.v[i][2][2]=sqrt(2.)*sin(Pi/4.*(2.*i-1.));
  }
  for(i=5;i<7;i++){
    vec.v[i][2][0]=0.;
    vec.v[i][2][1]=0.;
    vec.v[i][2][2]=pow(-1,i);
  }
  
  vec.v0[0]=0.;
  vec.v0[1]=0.;
  vec.v0[2]=0.;
  
  for(i=1;i<5;i++){
    for(p=0;p<3;p++){
      j=0;
      vec.e[i][p][0][0]=0.5*vec.v[(i+2)%4+1][p][0];
      vec.e[i][p][0][1]=0.5*vec.v[(i+2)%4+1][p][1];
      vec.e[i][p][0][2]=0.5*vec.v[(i+2)%4+1][p][2];
      
      j=1;
      vec.e[i][p][1][0]=0.5*vec.v[i%4+1][p][0];
      vec.e[i][p][1][1]=0.5*vec.v[i%4+1][p][1];
      vec.e[i][p][1][2]=0.5*vec.v[i%4+1][p][2];
    }
  }
  
  vec.e0[0]=0.;
  vec.e0[1]=0.;
  vec.e0[2]=0.;
  
  for(i=1;i<5;i++){
    for(p=0;p<3;p++){
      for(j=0;j<2;j++){
	vec.b[i][p][j][0]=(vec.v[i][p][1]*vec.e[i][p][j][2]-vec.v[i][p][2]*vec.e[i][p][j][1]);
	vec.b[i][p][j][1]=(vec.v[i][p][2]*vec.e[i][p][j][0]-vec.v[i][p][0]*vec.e[i][p][j][2]);
	vec.b[i][p][j][2]=(vec.v[i][p][0]*vec.e[i][p][j][1]-vec.v[i][p][1]*vec.e[i][p][j][0]);
      }
    }
  }
  
  vec.b0[0]=0.;
  vec.b0[1]=0.;
  vec.b0[2]=0.;

  cell=malloc(sizeof(Cell***)*2);

  for(t=0;t<2;t++){
    cell[t]=malloc(sizeof(Cell**)*Nx);
    for(k=0;k<Nx;k++){
      cell[t][k]=malloc(sizeof(Cell*)*Ny);
      for(l=0;l<Ny;l++){
	cell[t][k][l]=malloc(sizeof(Cell)*Nz);
	if(cell[t][k][l]==NULL){
	  fprintf(stderr,"Don't get MEMORY\n");
	  exit(-1);
	}
      }
    }
  }

  for(t=0;t<2;t++){
    for(k=0;k<Nx;k++){
      for(l=0;l<Ny;l++){
	for(m=0;m<Nz;m++){
	  x=(k-Nx/2-0.5);
	  y=(l-Ny/2-0.5);
	  z=(m-Nz/2-0.5);
	  r=sqrt(x*x+y*y+z*z);

	  cell[t][k][l][m].rho[0]=0.;
	  cell[t][k][l][m].rho[1]=0.;

	  for(d=0;d<3;d++){
	    cell[t][k][l][m].E[d]=0.;
	    cell[t][k][l][m].J[d]=0.;
	  }

	  cell[t][k][l][m].V[0][0]=V0x;
	  cell[t][k][l][m].V[0][1]=V0y;
	  cell[t][k][l][m].V[0][2]=V0z;
	  cell[t][k][l][m].V[1][0]=V0x;
	  cell[t][k][l][m].V[1][1]=V0y;
	  cell[t][k][l][m].V[1][2]=V0z;

	  /*cell[t][k][l][m].B0[0]=mu0*I*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5));
	  cell[t][k][l][m].B0[1]=mu0*I*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5));
	  cell[t][k][l][m].B0[2]=mu0*I*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5));*/

	  cell[t][k][l][m].B0[0]=mu0*I/2./Pi*(-(z+R*cos(alpha))
					      /(pow(x+R*sin(alpha),2)+pow(z+R*cos(alpha),2))
					      +(z-R*cos(alpha))
					      /(pow(x-R*sin(alpha),2)+pow(z-R*cos(alpha),2)));
	  cell[t][k][l][m].B0[1]=0.;
	  cell[t][k][l][m].B0[2]=-mu0*I/2./Pi*(-(x+R*sin(alpha))
					       /(pow(x+R*sin(alpha),2)+pow(z+R*cos(alpha),2))
					       +(x-R*sin(alpha))
					       /(pow(x-R*sin(alpha),2)+pow(z-R*cos(alpha),2)));

	  /*cell[t][k][l][m].B0[0]=integrate_bx(k,l,m);
	  cell[t][k][l][m].B0[1]=integrate_by(k,l,m);
	  cell[t][k][l][m].B0[2]=integrate_bz(k,l,m);*/

	  /*cell[t][k][l][m].B0[0]=0.;
	  cell[t][k][l][m].B0[1]=0.;
	  cell[t][k][l][m].B0[2]=0.;*/

	  cell[t][k][l][m].rho_c=0.;
	}
      }
    }
  }

  for(t=0;t<2;t++){
    for(k=0;k<Nx;k++){
      for(l=0;l<Ny;l++){
	for(m=0;m<Nz;m++){
	  Z[0][0]=(-q*N0*(cell[t][k][l][m].V[0][1]*(cell[t][k][l][m].B[2]+cell[t][k][l][m].B0[2])
			  -cell[t][k][l][m].V[0][2]*(cell[t][k][l][m].B[1]+cell[t][k][l][m].B0[1])));
	  Z[0][1]=(-q*N0*(cell[t][k][l][m].V[0][2]*(cell[t][k][l][m].B[0]+cell[t][k][l][m].B0[0])
			  -cell[t][k][l][m].V[0][0]*(cell[t][k][l][m].B[2]+cell[t][k][l][m].B0[2])));
	  Z[0][2]=(-q*N0*(cell[t][k][l][m].V[0][0]*(cell[t][k][l][m].B[1]+cell[t][k][l][m].B0[1])
			  -cell[t][k][l][m].V[0][1]*(cell[t][k][l][m].B[0]+cell[t][k][l][m].B0[0])));
	  
	  Z[1][0]=(+q*N0*(cell[t][k][l][m].V[1][1]*(cell[t][k][l][m].B[2]+cell[t][k][l][m].B0[2])
			  -cell[t][k][l][m].V[1][2]*(cell[t][k][l][m].B[1]+cell[t][k][l][m].B0[1])));
	  Z[1][1]=(+q*N0*(cell[t][k][l][m].V[1][2]*(cell[t][k][l][m].B[0]+cell[t][k][l][m].B0[0])
			  -cell[t][k][l][m].V[1][0]*(cell[t][k][l][m].B[2]+cell[t][k][l][m].B0[2])));
	  Z[1][2]=(+q*N0*(cell[t][k][l][m].V[1][0]*(cell[t][k][l][m].B[1]+cell[t][k][l][m].B0[1])
			  -cell[t][k][l][m].V[1][1]*(cell[t][k][l][m].B[0]+cell[t][k][l][m].B0[0])));
	  
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
	  
	  V[0][0]=cell[t][k][l][m].V[0][0]+F[0][0]/(2.*me*N0);
	  V[0][1]=cell[t][k][l][m].V[0][1]+F[0][1]/(2.*me*N0);
	  V[0][2]=cell[t][k][l][m].V[0][2]+F[0][2]/(2.*me*N0);
	  
	  V[1][0]=cell[t][k][l][m].V[1][0]+F[1][0]/(2.*mi*N0);
	  V[1][1]=cell[t][k][l][m].V[1][1]+F[1][1]/(2.*mi*N0);
	  V[1][2]=cell[t][k][l][m].V[1][2]+F[1][2]/(2.*mi*N0);

	  cell[t][k][l][m].iota[0]=N0*kb*Te*pow(me*N0,-gamma0);
	  cell[t][k][l][m].iota[1]=N0*kb*Ti*pow(mi*N0,-gamma0);
	  for(i=1;i<7;i++){
	    for(p=0;p<3;p++){
	      cell[t][k][l][m].f[0][i][p]=w[i]*me*N0*(3.*cell[t][k][l][m].iota[0]*pow(me*N0,gamma0-1.)
						      +3.*(vec.v[i][p][0]*V[0][0]+vec.v[i][p][1]*V[0][1]+vec.v[i][p][2]*V[0][2])
						      +4.5*pow((vec.v[i][p][0]*V[0][0]+vec.v[i][p][1]*V[0][1]+vec.v[i][p][2]*V[0][2]),2)
						      -1.5*(V[0][0]*V[0][0]+V[0][1]*V[0][1]+V[0][2]*V[0][2]));
	      cell[t][k][l][m].f[1][i][p]=w[i]*mi*N0*(3.*cell[t][k][l][m].iota[1]*pow(mi*N0,gamma0-1.)
						      +3.*(vec.v[i][p][0]*V[1][0]+vec.v[i][p][1]*V[1][1]+vec.v[i][p][2]*V[1][2])
						      +4.5*pow((vec.v[i][p][0]*V[1][0]+vec.v[i][p][1]*V[1][1]+vec.v[i][p][2]*V[1][2]),2)
						      -1.5*(V[1][0]*V[1][0]+V[1][1]*V[1][1]+V[1][2]*V[1][2]));
	      
	      cell[t][k][l][m].feq[0][i][p]=0.;
	      cell[t][k][l][m].feq[1][i][p]=0.;
	      
	      cell[t][k][l][m].T[0][i][p]=0.;
	      cell[t][k][l][m].T[1][i][p]=0.;
	    }
	  }
	  
	  cell[t][k][l][m].f0[0]=3.*w[0]*me*N0*(1.-0.5*(4.*cell[t][k][l][m].iota[0]*pow(me*N0,gamma0-1.)
							+(V[0][0]*V[0][0]+V[0][1]*V[0][1]+V[0][2]*V[0][2])));
	  cell[t][k][l][m].f0[1]=3.*w[0]*mi*N0*(1.-0.5*(4.*cell[t][k][l][m].iota[1]*pow(mi*N0,gamma0-1.)
							+(V[1][0]*V[1][0]+V[1][1]*V[1][1]+V[1][2]*V[1][2])));
	  
	  cell[t][k][l][m].feq0[0]=0.;
	  cell[t][k][l][m].feq0[1]=0.;
	  
	  cell[t][k][l][m].T0[0]=0.;
	  cell[t][k][l][m].T0[1]=0.;

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

		cell[t][k][l][m].g[i][p][j]=(E[0]*vec.e[i][p][j][0]+E[1]*vec.e[i][p][j][1]+E[2]*vec.e[i][p][j][2])/4.+(bx*vec.b[i][p][j][0]+by*vec.b[i][p][j][1]+bz*vec.b[i][p][j][2])/8.;
		cell[t][k][l][m].geq[i][p][j]=0.;
	      }
	    }
	  }

	  cell[t][k][l][m].g0=0.;
	  cell[t][k][l][m].geq0=0.;
	}
      }
    }
  }

  return(0);
}

int time_evolution(int t)
{
  cal_val();

  /* cell[tpp][Nx/2-1][Ny/2][Nz/2-1].J[2]+=I*sin(Pi/1000.*t)/4.;
  cell[tpp][Nx/2-1][Ny/2][Nz/2].J[2]+=I*sin(Pi/1000.*t)/4.;
  cell[tpp][Nx/2][Ny/2][Nz/2-1].J[2]+=I*sin(Pi/1000.*t)/4.;
  cell[tpp][Nx/2][Ny/2][Nz/2].J[2]+=I*sin(Pi/1000.*t)/4.;*/

  cal_T();

  step();

  tpp=tnx;
  tnx=(tnx+1)%2;

  return(0);
}

int cal_val()
{
  int k,l,m;
  int i,p,j,s,d;
  double bx,by,bz;

  for(k=0;k<Nx;k++){
    for(l=0;l<Ny;l++){
      for(m=0;m<Nz;m++){
	cell[tpp][k][l][m].rho[0]=cell[tpp][k][l][m].f0[0];
	cell[tpp][k][l][m].rho[1]=cell[tpp][k][l][m].f0[1];
	cell[tpp][k][l][m].rho_c=0.;

	for(d=0;d<3;d++){
	  cell[tpp][k][l][m].V[0][d]=0.;
	  cell[tpp][k][l][m].V[1][d]=0.;
	  cell[tpp][k][l][m].E[d]=0.;
	  cell[tpp][k][l][m].B[d]=0.;
	  cell[tpp][k][l][m].J[d]=0.;
	}

	for(s=0;s<2;s++){
	  for(i=1;i<7;i++){
	    for(p=0;p<3;p++){
	      cell[tpp][k][l][m].rho[s]+=cell[tpp][k][l][m].f[s][i][p];
	      
	      for(d=0;d<3;d++){
		cell[tpp][k][l][m].V[s][d]+=cell[tpp][k][l][m].f[s][i][p]*vec.v[i][p][d];
	      }
	    }
	  }
	}

	for(s=0;s<2;s++){
	  for(d=0;d<3;d++){
	    cell[tpp][k][l][m].V[s][d]=cell[tpp][k][l][m].V[s][d]/cell[tpp][k][l][m].rho[s];
	  }
	}


	for(i=1;i<5;i++){
	  for(p=0;p<3;p++){
	    for(j=0;j<2;j++){
	      for(d=0;d<3;d++){
		cell[tpp][k][l][m].E[d]+=cell[tpp][k][l][m].g[i][p][j]*vec.e[i][p][j][d];
		cell[tpp][k][l][m].B[d]+=cell[tpp][k][l][m].g[i][p][j]*vec.b[i][p][j][d];
	      }
	    }
	  }
	}

	for(d=0;d<3;d++){
	  cell[tpp][k][l][m].J[d]=-q/me*cell[tpp][k][l][m].rho[0]*cell[tpp][k][l][m].V[0][d]+q/mi*cell[tpp][k][l][m].rho[1]*cell[tpp][k][l][m].V[1][d];
	}

	cell[tpp][k][l][m].rho_c=-q/me*cell[tpp][k][l][m].rho[0]+q/mi*cell[tpp][k][l][m].rho[1];
      }
    }
  }

  return(0);
}

int cal_T()
{
  int k,l,m;
  int i,p,s,j;
  double Z[2][3];
  double F[2][3];
  double V[2][3];
  double E[3];
  double J[3];
  double temp[5];

  for(k=0;k<Nx;k++){
    for(l=0;l<Ny;l++){
      for(m=0;m<Nz;m++){
	Z[0][0]=(-q/me*cell[tpp][k][l][m].rho[0]*(cell[tpp][k][l][m].E[0]-0.25*mu0*cell[tpp][k][l][m].J[0]
						  +cell[tpp][k][l][m].V[0][1]*(cell[tpp][k][l][m].B[2]+cell[tpp][k][l][m].B0[2])
						  -cell[tpp][k][l][m].V[0][2]*(cell[tpp][k][l][m].B[1]+cell[tpp][k][l][m].B0[1]))
		 -nu*cell[tpp][k][l][m].rho[0]*(cell[tpp][k][l][m].V[0][0]-cell[tpp][k][l][m].V[1][0]));
	Z[0][1]=(-q/me*cell[tpp][k][l][m].rho[0]*(cell[tpp][k][l][m].E[1]-0.25*mu0*cell[tpp][k][l][m].J[1]
						  +cell[tpp][k][l][m].V[0][2]*(cell[tpp][k][l][m].B[0]+cell[tpp][k][l][m].B0[0])
						  -cell[tpp][k][l][m].V[0][0]*(cell[tpp][k][l][m].B[2]+cell[tpp][k][l][m].B0[2]))
		 -nu*cell[tpp][k][l][m].rho[0]*(cell[tpp][k][l][m].V[0][1]-cell[tpp][k][l][m].V[1][1]));
	Z[0][2]=(-q/me*cell[tpp][k][l][m].rho[0]*(cell[tpp][k][l][m].E[2]-0.25*mu0*cell[tpp][k][l][m].J[2]
						  +cell[tpp][k][l][m].V[0][0]*(cell[tpp][k][l][m].B[1]+cell[tpp][k][l][m].B0[1])
						  -cell[tpp][k][l][m].V[0][1]*(cell[tpp][k][l][m].B[0]+cell[tpp][k][l][m].B0[0]))
		 -nu*cell[tpp][k][l][m].rho[0]*(cell[tpp][k][l][m].V[0][2]-cell[tpp][k][l][m].V[1][2]));
	
	Z[1][0]=(+q/mi*cell[tpp][k][l][m].rho[0]*(cell[tpp][k][l][m].E[0]-0.25*mu0*cell[tpp][k][l][m].J[0]
						  +cell[tpp][k][l][m].V[1][1]*(cell[tpp][k][l][m].B[2]+cell[tpp][k][l][m].B0[2])
						  -cell[tpp][k][l][m].V[1][2]*(cell[tpp][k][l][m].B[1]+cell[tpp][k][l][m].B0[1]))
		 -nu*cell[tpp][k][l][m].rho[0]*(cell[tpp][k][l][m].V[1][0]-cell[tpp][k][l][m].V[0][0]));
	Z[1][1]=(+q/mi*cell[tpp][k][l][m].rho[0]*(cell[tpp][k][l][m].E[1]-0.25*mu0*cell[tpp][k][l][m].J[1]
						  +cell[tpp][k][l][m].V[1][2]*(cell[tpp][k][l][m].B[0]+cell[tpp][k][l][m].B0[0])
						  -cell[tpp][k][l][m].V[1][0]*(cell[tpp][k][l][m].B[2]+cell[tpp][k][l][m].B0[2]))
		 -nu*cell[tpp][k][l][m].rho[0]*(cell[tpp][k][l][m].V[1][1]-cell[tpp][k][l][m].V[0][1]));
	Z[1][2]=(+q/mi*cell[tpp][k][l][m].rho[0]*(cell[tpp][k][l][m].E[2]-0.25*mu0*cell[tpp][k][l][m].J[2]
						  +cell[tpp][k][l][m].V[1][0]*(cell[tpp][k][l][m].B[1]+cell[tpp][k][l][m].B0[1])
						  -cell[tpp][k][l][m].V[1][1]*(cell[tpp][k][l][m].B[0]+cell[tpp][k][l][m].B0[0]))
		 -nu*cell[tpp][k][l][m].rho[0]*(cell[tpp][k][l][m].V[1][2]-cell[tpp][k][l][m].V[0][2]));
	
	temp[0]=1.+mu0*q*q/(8.*mi*mi)*cell[tpp][k][l][m].rho[1];
	temp[1]=  -mu0*q*q/(8.*me*mi)*cell[tpp][k][l][m].rho[0];
	temp[2]=1.+mu0*q*q/(8.*me*me)*cell[tpp][k][l][m].rho[0];
	temp[3]=  -mu0*q*q/(8.*me*mi)*cell[tpp][k][l][m].rho[1];
	temp[4]=1.+0.125*mu0*q*q*(cell[tpp][k][l][m].rho[0]/(me*me)+cell[tpp][k][l][m].rho[1]/(mi*mi));
	
	
	F[0][0]=(temp[0]*Z[0][0]-temp[1]*Z[1][0])/temp[4];
	F[0][1]=(temp[0]*Z[0][1]-temp[1]*Z[1][1])/temp[4];
	F[0][2]=(temp[0]*Z[0][2]-temp[1]*Z[1][2])/temp[4];
	
	F[1][0]=(temp[2]*Z[1][0]-temp[3]*Z[0][0])/temp[4];
	F[1][1]=(temp[2]*Z[1][1]-temp[3]*Z[0][1])/temp[4];
	F[1][2]=(temp[2]*Z[1][2]-temp[3]*Z[0][2])/temp[4];
	
	V[0][0]=cell[tpp][k][l][m].V[0][0]+F[0][0]/(2.*cell[tpp][k][l][m].rho[0]);
	V[0][1]=cell[tpp][k][l][m].V[0][1]+F[0][1]/(2.*cell[tpp][k][l][m].rho[0]);
	V[0][2]=cell[tpp][k][l][m].V[0][2]+F[0][2]/(2.*cell[tpp][k][l][m].rho[0]);
	
	V[1][0]=cell[tpp][k][l][m].V[1][0]+F[1][0]/(2.*cell[tpp][k][l][m].rho[1]);
	V[1][1]=cell[tpp][k][l][m].V[1][1]+F[1][1]/(2.*cell[tpp][k][l][m].rho[1]);
	V[1][2]=cell[tpp][k][l][m].V[1][2]+F[1][2]/(2.*cell[tpp][k][l][m].rho[1]);
	
	for(s=0;s<2;s++){
	  for(i=1;i<7;i++){
	    for(p=0;p<3;p++){
	      cell[tpp][k][l][m].T[s][i][p]=(1.-omega[s]/2.)*w[i]*(3.*((vec.v[i][p][0]-V[s][0])*F[s][0]+
								       (vec.v[i][p][1]-V[s][1])*F[s][1]+
								       (vec.v[i][p][2]-V[s][2])*F[s][2])+
								   9.*(((vec.v[i][p][0]*V[s][0])+
									(vec.v[i][p][1]*V[s][1])+
									(vec.v[i][p][2]*V[s][2]))*
								       ((vec.v[i][p][0]*F[s][0])+
									(vec.v[i][p][1]*F[s][1])+
									(vec.v[i][p][2]*F[s][2]))));
	    }
	  }
	}
	
	cell[tpp][k][l][m].T0[0]=(1.-omega[0]/2.)*w[0]*(-3.*(V[0][0]*F[0][0]+
							     V[0][1]*F[0][1]+
							     V[0][2]*F[0][2]));
	cell[tpp][k][l][m].T0[1]=(1.-omega[1]/2.)*w[0]*(-3.*(V[1][0]*F[1][0]+
							     V[1][1]*F[1][1]+
							     V[1][2]*F[1][2]));

	for(s=0;s<2;s++){
	  for(i=1;i<7;i++){
	    for(p=0;p<3;p++){
	      cell[tpp][k][l][m].feq[s][i][p]=w[i]*cell[tpp][k][l][m].rho[s]
		*(3.*cell[tpp][k][l][m].iota[s]*pow(cell[tpp][k][l][m].rho[s],gamma0-1.)
		  +3.*((vec.v[i][p][0]*V[s][0])+(vec.v[i][p][1]*V[s][1])+(vec.v[i][p][2]*V[s][2]))
		  +4.5*pow((vec.v[i][p][0]*V[s][0])+(vec.v[i][p][1]*V[s][1])+(vec.v[i][p][2]*V[s][2]),2.)
		  -1.5*(V[s][0]*V[s][0]+V[s][1]*V[s][1]+V[s][2]*V[s][2]));
	    }
	  }
	}
	
	cell[tpp][k][l][m].feq0[0]=3.*w[0]*cell[tpp][k][l][m].rho[0]
	  *(1.-0.5*(4.*cell[tpp][k][l][m].iota[0]*pow(cell[tpp][k][l][m].rho[0],gamma0-1.)
		    +(V[0][0]*V[0][0]+V[0][1]*V[0][1]+V[0][2]*V[0][2])));

	cell[tpp][k][l][m].feq0[1]=3.*w[0]*cell[tpp][k][l][m].rho[1]
	  *(1.-0.5*(4.*cell[tpp][k][l][m].iota[1]*pow(cell[tpp][k][l][m].rho[1],gamma0-1.)
		    +(V[1][0]*V[1][0]+V[1][1]*V[1][1]+V[1][2]*V[1][2])));
	
	J[0]=cell[tpp][k][l][m].J[0]+0.5*(-q*F[0][0]/me+q*F[1][0]/mi);
	J[1]=cell[tpp][k][l][m].J[1]+0.5*(-q*F[0][1]/me+q*F[1][1]/mi);
	J[2]=cell[tpp][k][l][m].J[2]+0.5*(-q*F[0][2]/me+q*F[1][2]/mi);

	E[0]=cell[tpp][k][l][m].E[0]-0.25*mu0*J[0];
	E[1]=cell[tpp][k][l][m].E[1]-0.25*mu0*J[1];
	E[2]=cell[tpp][k][l][m].E[2]-0.25*mu0*J[2];

	for(i=1;i<5;i++){
	  for(p=0;p<3;p++){
	    for(j=0;j<2;j++){
	      cell[tpp][k][l][m].geq[i][p][j]=0.25*(E[0]*vec.e[i][p][j][0]+
						    E[1]*vec.e[i][p][j][1]+
						    E[2]*vec.e[i][p][j][2])
		+0.125*(cell[tpp][k][l][m].B[0]*vec.b[i][p][j][0]+
			cell[tpp][k][l][m].B[1]*vec.b[i][p][j][1]+
			cell[tpp][k][l][m].B[2]*vec.b[i][p][j][2]);
	    }
	  }
	}

	cell[tpp][k][l][m].geq0=0.;
      }
    }
  }
  
  return(0);
}

int step()
{
  int k,l,m;
  int kp,km,lp,lm,mp,mm;
  int s;

  for(k=0;k<Nx;k++){
    for(l=0;l<Ny;l++){
      for(m=0;m<Nz;m++){
	kp=(k+1)%Nx; 
	km=(k-1+Nx)%Nx;
	lp=(l+1)%Ny; 
	lm=(l-1+Ny)%Ny;
	mp=(m+1)%Nz; 
	mm=(m-1+Nz)%Nz;

	for(s=0;s<2;s++){
	  cell[tnx][k][l][m].f0[s]=cell[tpp][k][l][m].f0[s]-omega[s]*(cell[tpp][k][l][m].f0[s]-cell[tpp][k][l][m].feq0[s])+cell[tpp][k][l][m].T0[s];

	  cell[tnx][kp][lp][m].f[s][1][0]=cell[tpp][k][l][m].f[s][1][0]-omega[s]*(cell[tpp][k][l][m].f[s][1][0]-cell[tpp][k][l][m].feq[s][1][0])+cell[tpp][k][l][m].T[s][1][0];
	  cell[tnx][km][lp][m].f[s][2][0]=cell[tpp][k][l][m].f[s][2][0]-omega[s]*(cell[tpp][k][l][m].f[s][2][0]-cell[tpp][k][l][m].feq[s][2][0])+cell[tpp][k][l][m].T[s][2][0];
	  cell[tnx][km][lm][m].f[s][3][0]=cell[tpp][k][l][m].f[s][3][0]-omega[s]*(cell[tpp][k][l][m].f[s][3][0]-cell[tpp][k][l][m].feq[s][3][0])+cell[tpp][k][l][m].T[s][3][0];
	  cell[tnx][kp][lm][m].f[s][4][0]=cell[tpp][k][l][m].f[s][4][0]-omega[s]*(cell[tpp][k][l][m].f[s][4][0]-cell[tpp][k][l][m].feq[s][4][0])+cell[tpp][k][l][m].T[s][4][0];
	  cell[tnx][km][l][m].f[s][5][0] =cell[tpp][k][l][m].f[s][5][0]-omega[s]*(cell[tpp][k][l][m].f[s][5][0]-cell[tpp][k][l][m].feq[s][5][0])+cell[tpp][k][l][m].T[s][5][0];
	  cell[tnx][kp][l][m].f[s][6][0] =cell[tpp][k][l][m].f[s][6][0]-omega[s]*(cell[tpp][k][l][m].f[s][6][0]-cell[tpp][k][l][m].feq[s][6][0])+cell[tpp][k][l][m].T[s][6][0];

	  cell[tnx][kp][l][mp].f[s][1][1]=cell[tpp][k][l][m].f[s][1][1]-omega[s]*(cell[tpp][k][l][m].f[s][1][1]-cell[tpp][k][l][m].feq[s][1][1])+cell[tpp][k][l][m].T[s][1][1];
	  cell[tnx][km][l][mp].f[s][2][1]=cell[tpp][k][l][m].f[s][2][1]-omega[s]*(cell[tpp][k][l][m].f[s][2][1]-cell[tpp][k][l][m].feq[s][2][1])+cell[tpp][k][l][m].T[s][2][1];
	  cell[tnx][km][l][mm].f[s][3][1]=cell[tpp][k][l][m].f[s][3][1]-omega[s]*(cell[tpp][k][l][m].f[s][3][1]-cell[tpp][k][l][m].feq[s][3][1])+cell[tpp][k][l][m].T[s][3][1];
	  cell[tnx][kp][l][mm].f[s][4][1]=cell[tpp][k][l][m].f[s][4][1]-omega[s]*(cell[tpp][k][l][m].f[s][4][1]-cell[tpp][k][l][m].feq[s][4][1])+cell[tpp][k][l][m].T[s][4][1];
	  cell[tnx][k][lm][m].f[s][5][1] =cell[tpp][k][l][m].f[s][5][1]-omega[s]*(cell[tpp][k][l][m].f[s][5][1]-cell[tpp][k][l][m].feq[s][5][1])+cell[tpp][k][l][m].T[s][5][1];
	  cell[tnx][k][lp][m].f[s][6][1] =cell[tpp][k][l][m].f[s][6][1]-omega[s]*(cell[tpp][k][l][m].f[s][6][1]-cell[tpp][k][l][m].feq[s][6][1])+cell[tpp][k][l][m].T[s][6][1];

	  cell[tnx][k][lp][mp].f[s][1][2]=cell[tpp][k][l][m].f[s][1][2]-omega[s]*(cell[tpp][k][l][m].f[s][1][2]-cell[tpp][k][l][m].feq[s][1][2])+cell[tpp][k][l][m].T[s][1][2];
	  cell[tnx][k][lm][mp].f[s][2][2]=cell[tpp][k][l][m].f[s][2][2]-omega[s]*(cell[tpp][k][l][m].f[s][2][2]-cell[tpp][k][l][m].feq[s][2][2])+cell[tpp][k][l][m].T[s][2][2];
	  cell[tnx][k][lm][mm].f[s][3][2]=cell[tpp][k][l][m].f[s][3][2]-omega[s]*(cell[tpp][k][l][m].f[s][3][2]-cell[tpp][k][l][m].feq[s][3][2])+cell[tpp][k][l][m].T[s][3][2];
	  cell[tnx][k][lp][mm].f[s][4][2]=cell[tpp][k][l][m].f[s][4][2]-omega[s]*(cell[tpp][k][l][m].f[s][4][2]-cell[tpp][k][l][m].feq[s][4][2])+cell[tpp][k][l][m].T[s][4][2];
	  cell[tnx][k][l][mm].f[s][5][2] =cell[tpp][k][l][m].f[s][5][2]-omega[s]*(cell[tpp][k][l][m].f[s][5][2]-cell[tpp][k][l][m].feq[s][5][2])+cell[tpp][k][l][m].T[s][5][2];
	  cell[tnx][k][l][mp].f[s][6][2] =cell[tpp][k][l][m].f[s][6][2]-omega[s]*(cell[tpp][k][l][m].f[s][6][2]-cell[tpp][k][l][m].feq[s][6][2])+cell[tpp][k][l][m].T[s][6][2];
	}

	cell[tnx][k][l][m].g0=cell[tpp][k][l][m].g0-omega[2]*(cell[tpp][k][l][m].g0-cell[tpp][k][l][m].geq0);

	cell[tnx][kp][lp][m].g[1][0][0]=cell[tpp][k][l][m].g[1][0][0]-omega[2]*(cell[tpp][k][l][m].g[1][0][0]-cell[tpp][k][l][m].geq[1][0][0]);
	cell[tnx][km][lp][m].g[2][0][0]=cell[tpp][k][l][m].g[2][0][0]-omega[2]*(cell[tpp][k][l][m].g[2][0][0]-cell[tpp][k][l][m].geq[2][0][0]);
	cell[tnx][km][lm][m].g[3][0][0]=cell[tpp][k][l][m].g[3][0][0]-omega[2]*(cell[tpp][k][l][m].g[3][0][0]-cell[tpp][k][l][m].geq[3][0][0]);
	cell[tnx][kp][lm][m].g[4][0][0]=cell[tpp][k][l][m].g[4][0][0]-omega[2]*(cell[tpp][k][l][m].g[4][0][0]-cell[tpp][k][l][m].geq[4][0][0]);

	cell[tnx][kp][lp][m].g[1][0][1]=cell[tpp][k][l][m].g[1][0][1]-omega[2]*(cell[tpp][k][l][m].g[1][0][1]-cell[tpp][k][l][m].geq[1][0][1]);
	cell[tnx][km][lp][m].g[2][0][1]=cell[tpp][k][l][m].g[2][0][1]-omega[2]*(cell[tpp][k][l][m].g[2][0][1]-cell[tpp][k][l][m].geq[2][0][1]);
	cell[tnx][km][lm][m].g[3][0][1]=cell[tpp][k][l][m].g[3][0][1]-omega[2]*(cell[tpp][k][l][m].g[3][0][1]-cell[tpp][k][l][m].geq[3][0][1]);
	cell[tnx][kp][lm][m].g[4][0][1]=cell[tpp][k][l][m].g[4][0][1]-omega[2]*(cell[tpp][k][l][m].g[4][0][1]-cell[tpp][k][l][m].geq[4][0][1]);

	cell[tnx][kp][l][mp].g[1][1][0]=cell[tpp][k][l][m].g[1][1][0]-omega[2]*(cell[tpp][k][l][m].g[1][1][0]-cell[tpp][k][l][m].geq[1][1][0]);
	cell[tnx][km][l][mp].g[2][1][0]=cell[tpp][k][l][m].g[2][1][0]-omega[2]*(cell[tpp][k][l][m].g[2][1][0]-cell[tpp][k][l][m].geq[2][1][0]);
	cell[tnx][km][l][mm].g[3][1][0]=cell[tpp][k][l][m].g[3][1][0]-omega[2]*(cell[tpp][k][l][m].g[3][1][0]-cell[tpp][k][l][m].geq[3][1][0]);
	cell[tnx][kp][l][mm].g[4][1][0]=cell[tpp][k][l][m].g[4][1][0]-omega[2]*(cell[tpp][k][l][m].g[4][1][0]-cell[tpp][k][l][m].geq[4][1][0]);

	cell[tnx][kp][l][mp].g[1][1][1]=cell[tpp][k][l][m].g[1][1][1]-omega[2]*(cell[tpp][k][l][m].g[1][1][1]-cell[tpp][k][l][m].geq[1][1][1]);
	cell[tnx][km][l][mp].g[2][1][1]=cell[tpp][k][l][m].g[2][1][1]-omega[2]*(cell[tpp][k][l][m].g[2][1][1]-cell[tpp][k][l][m].geq[2][1][1]);
	cell[tnx][km][l][mm].g[3][1][1]=cell[tpp][k][l][m].g[3][1][1]-omega[2]*(cell[tpp][k][l][m].g[3][1][1]-cell[tpp][k][l][m].geq[3][1][1]);
	cell[tnx][kp][l][mm].g[4][1][1]=cell[tpp][k][l][m].g[4][1][1]-omega[2]*(cell[tpp][k][l][m].g[4][1][1]-cell[tpp][k][l][m].geq[4][1][1]);

	cell[tnx][k][lp][mp].g[1][2][0]=cell[tpp][k][l][m].g[1][2][0]-omega[2]*(cell[tpp][k][l][m].g[1][2][0]-cell[tpp][k][l][m].geq[1][2][0]);
	cell[tnx][k][lm][mp].g[2][2][0]=cell[tpp][k][l][m].g[2][2][0]-omega[2]*(cell[tpp][k][l][m].g[2][2][0]-cell[tpp][k][l][m].geq[2][2][0]);
	cell[tnx][k][lm][mm].g[3][2][0]=cell[tpp][k][l][m].g[3][2][0]-omega[2]*(cell[tpp][k][l][m].g[3][2][0]-cell[tpp][k][l][m].geq[3][2][0]);
	cell[tnx][k][lp][mm].g[4][2][0]=cell[tpp][k][l][m].g[4][2][0]-omega[2]*(cell[tpp][k][l][m].g[4][2][0]-cell[tpp][k][l][m].geq[4][2][0]);

	cell[tnx][k][lp][mp].g[1][2][1]=cell[tpp][k][l][m].g[1][2][1]-omega[2]*(cell[tpp][k][l][m].g[1][2][1]-cell[tpp][k][l][m].geq[1][2][1]);
	cell[tnx][k][lm][mp].g[2][2][1]=cell[tpp][k][l][m].g[2][2][1]-omega[2]*(cell[tpp][k][l][m].g[2][2][1]-cell[tpp][k][l][m].geq[2][2][1]);
	cell[tnx][k][lm][mm].g[3][2][1]=cell[tpp][k][l][m].g[3][2][1]-omega[2]*(cell[tpp][k][l][m].g[3][2][1]-cell[tpp][k][l][m].geq[3][2][1]);
	cell[tnx][k][lp][mm].g[4][2][1]=cell[tpp][k][l][m].g[4][2][1]-omega[2]*(cell[tpp][k][l][m].g[4][2][1]-cell[tpp][k][l][m].geq[4][2][1]);
      }
    }
  }

  return(0);
}

/*******************************************************************
magnetic field by coil
 bx
******************************************************************/
double integrate_bx(int k,int l,int m)
{
  gsl_integration_workspace *ws;
  double result,error;
  gsl_function F;
  klm p;

  p.k=k;
  p.l=l;
  p.m=m;

  ws=gsl_integration_workspace_alloc(10000);
  F.function=&fbx;
  F.params=&p;

  gsl_integration_qags(&F,0.,2*Pi,0.000001,0.000001,10000,ws,&result,&error);

  gsl_integration_workspace_free(ws);

  return(mu0*I/4./Pi*result);
}

/*******************************************************************
magnetic field by coil
 by
******************************************************************/
double integrate_by(int k,int l,int m)
{
  gsl_integration_workspace *ws;
  double result,error;
  gsl_function F;
  klm p;

  p.k=k;
  p.l=l;
  p.m=m;

  ws=gsl_integration_workspace_alloc(10000);
  F.function=&fby;
  F.params=&p;

  gsl_integration_qags(&F,0.,2*Pi,0.000001,0.000001,10000,ws,&result,&error);

  gsl_integration_workspace_free(ws);

  return(mu0*I/4./Pi*result);
}

/*******************************************************************
magnetic field by coil
 bz
******************************************************************/
double integrate_bz(int k,int l,int m)
{
  gsl_integration_workspace *ws;
  double result,error;
  gsl_function F;
  klm p;

  p.k=k;
  p.l=l;
  p.m=m;

  ws=gsl_integration_workspace_alloc(10000);
  F.function=&fbz;
  F.params=&p;

  gsl_integration_qags(&F,0.,2*Pi,0.000001,0.000001,10000,ws,&result,&error);

  gsl_integration_workspace_free(ws);

  return(mu0*I/4./Pi*result);
}

/*******************************************************************
function integrated by integrate_bx()
******************************************************************/
double fbx(double w,void *params)
{
  double r;
  klm *p=(klm *)params;
  double x;
  double y;
  double z;

  x=(double)(p->k)-Nx/2.;
  y=(double)(p->l)-Ny/2.;
  z=(double)(p->m)-Nz/2.;

  r=R*(z*cos(w)*cos(alpha)-y*cos(w)*sin(alpha))
    *pow(gsl_pow_2(R)+gsl_pow_2(x)+gsl_pow_2(y)+gsl_pow_2(z)
	 -2*R*(x*cos(w)+y*sin(w)*cos(alpha)+z*sin(w)*sin(alpha)),-1.5);
  
  return(r);
}

/*******************************************************************
function integrated by integrate_by()
******************************************************************/  
double fby(double w,void *params)
{
  double r;
  klm *p=(klm *)params;
  double x;
  double y;
  double z;

  x=(double)(p->k)-Nx/2.;
  y=(double)(p->l)-Ny/2.;
  z=(double)(p->m)-Nz/2.;

  r=R*(z*sin(w)+x*cos(w)*sin(alpha)-R*sin(alpha))
    *pow(gsl_pow_2(R)+gsl_pow_2(x)+gsl_pow_2(y)+gsl_pow_2(z)
	 -2*R*(x*cos(w)+y*sin(w)*cos(alpha)+z*sin(w)*sin(alpha)),-1.5);

  return(r);
}

/*******************************************************************
function integrated by integrate_bz()
******************************************************************/
double fbz(double w,void *params)
{
  double r;
  klm *p=(klm *)params;
  double x;
  double y;
  double z;

  x=(double)(p->k)-Nx/2.;
  y=(double)(p->l)-Ny/2.;
  z=(double)(p->m)-Nz/2.;

  r=R*(R*cos(alpha)-y*sin(w)-x*cos(w)*cos(alpha))
    *pow(gsl_pow_2(R)+gsl_pow_2(x)+gsl_pow_2(y)+gsl_pow_2(z)
	 -2*R*(x*cos(w)+y*sin(w)*cos(alpha)+z*sin(w)*sin(alpha)),-1.5);

  return(r);
}

int output(int c)
{
  int k,l,m;
  FILE *fp1;
  char filename[256];
  
  sprintf(filename,"lbm%03d-%d.dat",version,c);
  fp1=fopen(filename,"w");
  
  fprintf(fp1,"VARIABLES = \"X[m]\" \"Y[m]\" \"Z[m]\" \"Ne[/m3]\" \"Ni[/m3]\" \"Pe[N/m2]\" \"Pi[N/m2]\" \"Vx_e[m/s]\" \"Vy_e[m/s]\"  \"Vz_e[m/s]\" \"Vx_i[m/s]\"  \"Vy_i[m/s]\" \"Vz_i[m/s]\" \"rhoVx_e[]\" \"rhoVy_e[m/s]\"  \"rhoVz_e[m/s]\" \"rhoVx_i[m/s]\"  \"rhoVy_i[m/s]\" \"rhoVz_i[m/s]\"\"Ex[V/m]\" \"Ey[V/m]\" \"Ez[V/m]\" \"Bx[T]\" \"By[T]\" \"Bz[T]\" \"Bpx[T]\" \"Bpy[T]\" \"Bpz[T]\" \"B[T]\" \"Jx[A/m2]\" \"Jy[A/m2]\" \"Jz[A/m2]\" \n");
  fprintf(fp1,"ZONE T=\"STP:%d\", STRANDID=1, SOLUTIONTIME=%d, I=%d, J=%d, K=%d\n",c,c,Nz,Ny,Nx);
  
  for(k=0;k<Nx;k++){
    for(l=0;l<Ny;l++){
      for(m=0;m<Nz;m++){
	fprintf(fp1,"%.3E %.3E %.3E %.8E %.8E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E  %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E\n",
		k*dx,l*dx,m*dx,
		cell[tnx][k][l][m].rho[0]/me/(dx*dx*dx),
		cell[tnx][k][l][m].rho[1]/mi/(dx*dx*dx),
		cell[tnx][k][l][m].iota[0]*pow(cell[tnx][k][l][m].rho[0],gamma0)/(dx*dt*dt),
		cell[tnx][k][l][m].iota[0]*pow(cell[tnx][k][l][m].rho[0],gamma0)/(dx*dt*dt),
		cell[tnx][k][l][m].V[0][0]*dx/dt,
		cell[tnx][k][l][m].V[0][1]*dx/dt,
		cell[tnx][k][l][m].V[0][2]*dx/dt,
		cell[tnx][k][l][m].V[1][0]*dx/dt,
		cell[tnx][k][l][m].V[1][1]*dx/dt,
		cell[tnx][k][l][m].V[1][2]*dx/dt,
		cell[tnx][k][l][m].rho[0]/(dx*dx*dx)*cell[tnx][k][l][m].V[0][0]*dx/dt,
		cell[tnx][k][l][m].rho[0]/(dx*dx*dx)*cell[tnx][k][l][m].V[0][1]*dx/dt,
		cell[tnx][k][l][m].rho[0]/(dx*dx*dx)*cell[tnx][k][l][m].V[0][2]*dx/dt,
		cell[tnx][k][l][m].rho[1]/(dx*dx*dx)*cell[tnx][k][l][m].V[1][0]*dx/dt,
		cell[tnx][k][l][m].rho[1]/(dx*dx*dx)*cell[tnx][k][l][m].V[1][1]*dx/dt,
		cell[tnx][k][l][m].rho[1]/(dx*dx*dx)*cell[tnx][k][l][m].V[1][2]*dx/dt,
		cell[tnx][k][l][m].E[0]*dx/(dt*dt*dt),
		cell[tnx][k][l][m].E[1]*dx/(dt*dt*dt),
		cell[tnx][k][l][m].E[2]*dx/(dt*dt*dt),
		(cell[tnx][k][l][m].B0[0]+cell[tnx][k][l][m].B[0])/(dt*dt),
		(cell[tnx][k][l][m].B0[1]+cell[tnx][k][l][m].B[1])/(dt*dt),
		(cell[tnx][k][l][m].B0[2]+cell[tnx][k][l][m].B[2])/(dt*dt),
		cell[tnx][k][l][m].B[0]/(dt*dt),
		cell[tnx][k][l][m].B[1]/(dt*dt),
		cell[tnx][k][l][m].B[2]/(dt*dt),
		sqrt(pow(cell[tnx][k][l][m].B0[0]+cell[tnx][k][l][m].B[0],2)+
		     pow(cell[tnx][k][l][m].B0[1]+cell[tnx][k][l][m].B[1],2)+
		     pow(cell[tnx][k][l][m].B0[2]+cell[tnx][k][l][m].B[2],2))/(dt*dt),
		cell[tnx][k][l][m].J[0]/(dx*dx),
		cell[tnx][k][l][m].J[1]/(dx*dx),
		cell[tnx][k][l][m].J[2]/(dx*dx));
	
      }
      fprintf(fp1,"\n");
    }
  }
  
  fclose(fp1);

  return(0);
}

int output_gnuplot(int c)
{
  int k,l,m;
  FILE *fp1;
  char filename[256];
  
  sprintf(filename,"lbm%03d-%d_gnuplot.dat",version,c);
  fp1=fopen(filename,"w");
  
  l=0;
  for(k=0;k<Nx;k++){
    for(m=0;m<Nz;m++){
      fprintf(fp1,"%.3E %.3E %.3E %.8E %.8E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E  %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E\n",
	      k*dx,l*dx,m*dx,
	      cell[tnx][k][l][m].rho[0]/me/(dx*dx*dx),
	      cell[tnx][k][l][m].rho[1]/mi/(dx*dx*dx),
	      cell[tnx][k][l][m].iota[0]*pow(cell[tnx][k][l][m].rho[0],gamma0)/(dx*dt*dt),
	      cell[tnx][k][l][m].iota[0]*pow(cell[tnx][k][l][m].rho[0],gamma0)/(dx*dt*dt),
	      cell[tnx][k][l][m].V[0][0]*dx/dt,
	      cell[tnx][k][l][m].V[0][1]*dx/dt,
	      cell[tnx][k][l][m].V[0][2]*dx/dt,
	      cell[tnx][k][l][m].V[1][0]*dx/dt,
	      cell[tnx][k][l][m].V[1][1]*dx/dt,
	      cell[tnx][k][l][m].V[1][2]*dx/dt,
	      cell[tnx][k][l][m].rho[0]/(dx*dx*dx)*cell[tnx][k][l][m].V[0][0]*dx/dt,
	      cell[tnx][k][l][m].rho[0]/(dx*dx*dx)*cell[tnx][k][l][m].V[0][1]*dx/dt,
	      cell[tnx][k][l][m].rho[0]/(dx*dx*dx)*cell[tnx][k][l][m].V[0][2]*dx/dt,
	      cell[tnx][k][l][m].rho[1]/(dx*dx*dx)*cell[tnx][k][l][m].V[1][0]*dx/dt,
	      cell[tnx][k][l][m].rho[1]/(dx*dx*dx)*cell[tnx][k][l][m].V[1][1]*dx/dt,
	      cell[tnx][k][l][m].rho[1]/(dx*dx*dx)*cell[tnx][k][l][m].V[1][2]*dx/dt,
	      cell[tnx][k][l][m].E[0]*dx/(dt*dt*dt),
	      cell[tnx][k][l][m].E[1]*dx/(dt*dt*dt),
	      cell[tnx][k][l][m].E[2]*dx/(dt*dt*dt),
	      (cell[tnx][k][l][m].B0[0]+cell[tnx][k][l][m].B[0])/(dt*dt),
	      (cell[tnx][k][l][m].B0[1]+cell[tnx][k][l][m].B[1])/(dt*dt),
	      (cell[tnx][k][l][m].B0[2]+cell[tnx][k][l][m].B[2])/(dt*dt),
	      cell[tnx][k][l][m].B[0]/(dt*dt),
	      cell[tnx][k][l][m].B[1]/(dt*dt),
	      cell[tnx][k][l][m].B[2]/(dt*dt),
	      sqrt(pow(cell[tnx][k][l][m].B0[0]+cell[tnx][k][l][m].B[0],2)+
		   pow(cell[tnx][k][l][m].B0[1]+cell[tnx][k][l][m].B[1],2)+
		   pow(cell[tnx][k][l][m].B0[2]+cell[tnx][k][l][m].B[2],2))/(dt*dt),
	      cell[tnx][k][l][m].J[0]/(dx*dx),
	      cell[tnx][k][l][m].J[1]/(dx*dx),
	      cell[tnx][k][l][m].J[2]/(dx*dx));
      
    }
    fprintf(fp1,"\n");
  }
  
  fclose(fp1);

  return(0);
}

int dump()
{
  int k,l,m;
  int i,p,j;
  double d;

  for(k=0;k<Nx;k++){
    for(l=0;l<Ny;l++){
      for(m=0;m<Nz;m++){
	if(k==0||k==Nx-1||m==0||m==Nz-1){
	  cell[t][k][l][m].V[0][0]=V0x;
	  cell[t][k][l][m].V[0][1]=V0y;
	  cell[t][k][l][m].V[0][2]=V0z;
	  cell[t][k][l][m].V[1][0]=V0x;
	  cell[t][k][l][m].V[1][1]=V0y;
	  cell[t][k][l][m].V[1][2]=V0z;

	  Z[0][0]=(-q*N0*(cell[t][k][l][m].V[0][1]*(cell[t][k][l][m].B[2]+cell[t][k][l][m].B0[2])
			  -cell[t][k][l][m].V[0][2]*(cell[t][k][l][m].B[1]+cell[t][k][l][m].B0[1])));
	  Z[0][1]=(-q*N0*(cell[t][k][l][m].V[0][2]*(cell[t][k][l][m].B[0]+cell[t][k][l][m].B0[0])
			  -cell[t][k][l][m].V[0][0]*(cell[t][k][l][m].B[2]+cell[t][k][l][m].B0[2])));
	  Z[0][2]=(-q*N0*(cell[t][k][l][m].V[0][0]*(cell[t][k][l][m].B[1]+cell[t][k][l][m].B0[1])
			  -cell[t][k][l][m].V[0][1]*(cell[t][k][l][m].B[0]+cell[t][k][l][m].B0[0])));
	  
	  Z[1][0]=(+q*N0*(cell[t][k][l][m].V[1][1]*(cell[t][k][l][m].B[2]+cell[t][k][l][m].B0[2])
			  -cell[t][k][l][m].V[1][2]*(cell[t][k][l][m].B[1]+cell[t][k][l][m].B0[1])));
	  Z[1][1]=(+q*N0*(cell[t][k][l][m].V[1][2]*(cell[t][k][l][m].B[0]+cell[t][k][l][m].B0[0])
			  -cell[t][k][l][m].V[1][0]*(cell[t][k][l][m].B[2]+cell[t][k][l][m].B0[2])));
	  Z[1][2]=(+q*N0*(cell[t][k][l][m].V[1][0]*(cell[t][k][l][m].B[1]+cell[t][k][l][m].B0[1])
			  -cell[t][k][l][m].V[1][1]*(cell[t][k][l][m].B[0]+cell[t][k][l][m].B0[0])));
	  
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
	  
	  V[0][0]=cell[t][k][l][m].V[0][0]+F[0][0]/(2.*me*N0);
	  V[0][1]=cell[t][k][l][m].V[0][1]+F[0][1]/(2.*me*N0);
	  V[0][2]=cell[t][k][l][m].V[0][2]+F[0][2]/(2.*me*N0);
	  
	  V[1][0]=cell[t][k][l][m].V[1][0]+F[1][0]/(2.*mi*N0);
	  V[1][1]=cell[t][k][l][m].V[1][1]+F[1][1]/(2.*mi*N0);
	  V[1][2]=cell[t][k][l][m].V[1][2]+F[1][2]/(2.*mi*N0);

	  cell[t][k][l][m].iota[0]=N0*kb*Te*pow(me*N0,-gamma0);
	  cell[t][k][l][m].iota[1]=N0*kb*Ti*pow(mi*N0,-gamma0);

	  for(i=1;i<7;i++){
	    for(p=0;p<3;p++){
	      cell[t][k][l][m].f[0][i][p]=w[i]*me*N0*(3.*cell[t][k][l][m].iota[0]*pow(me*N0,gamma0-1.)
						      +3.*(vec.v[i][p][0]*V[0][0]+vec.v[i][p][1]*V[0][1]+vec.v[i][p][2]*V[0][2])
						      +4.5*pow((vec.v[i][p][0]*V[0][0]+vec.v[i][p][1]*V[0][1]+vec.v[i][p][2]*V[0][2]),2)
						      -1.5*(V[0][0]*V[0][0]+V[0][1]*V[0][1]+V[0][2]*V[0][2]));
	      cell[t][k][l][m].f[1][i][p]=w[i]*mi*N0*(3.*cell[t][k][l][m].iota[1]*pow(mi*N0,gamma0-1.)
						      +3.*(vec.v[i][p][0]*V[1][0]+vec.v[i][p][1]*V[1][1]+vec.v[i][p][2]*V[1][2])
						      +4.5*pow((vec.v[i][p][0]*V[1][0]+vec.v[i][p][1]*V[1][1]+vec.v[i][p][2]*V[1][2]),2)
						      -1.5*(V[1][0]*V[1][0]+V[1][1]*V[1][1]+V[1][2]*V[1][2]));
	      
	      cell[t][k][l][m].feq[0][i][p]=0.;
	      cell[t][k][l][m].feq[1][i][p]=0.;
	      
	      cell[t][k][l][m].T[0][i][p]=0.;
	      cell[t][k][l][m].T[1][i][p]=0.;
	    }
	  }
	  
	  cell[t][k][l][m].f0[0]=3.*w[0]*me*N0*(1.-0.5*(4.*cell[t][k][l][m].iota[0]*pow(me*N0,gamma0-1.)
							+(V[0][0]*V[0][0]+V[0][1]*V[0][1]+V[0][2]*V[0][2])));
	  cell[t][k][l][m].f0[1]=3.*w[0]*mi*N0*(1.-0.5*(4.*cell[t][k][l][m].iota[1]*pow(mi*N0,gamma0-1.)
							+(V[1][0]*V[1][0]+V[1][1]*V[1][1]+V[1][2]*V[1][2])));
	  
	  cell[t][k][l][m].feq0[0]=0.;
	  cell[t][k][l][m].feq0[1]=0.;
	  
	  cell[t][k][l][m].T0[0]=0.;
	  cell[t][k][l][m].T0[1]=0.;
	}

	/*	d=1.;

	if(k<Dump){
	  d=d*pow((double)k/Dump,4.);
	}else if(k>Nx-1-Dump){
	  d=d*pow((double)(Nx-1-k)/Dump,4.);
	}

	if(m<Dump){
	  d=d*pow((double)m/Dump,4.);
	}else if(m>Nz-1-Dump){
	  d=d*pow((double)(Nz-1-m)/Dump,4.);
	}

	if(d<1.){
	  //fprintf(stderr,"%f\n",d);
	  cell[tpp][k][l][m].E[0]=d*cell[tpp][k][l][m].E[0];
	  cell[tpp][k][l][m].E[1]=d*cell[tpp][k][l][m].E[1];
	  cell[tpp][k][l][m].E[2]=d*cell[tpp][k][l][m].E[2];
	  cell[tpp][k][l][m].B[0]=d*cell[tpp][k][l][m].B[0];
	  cell[tpp][k][l][m].B[1]=d*cell[tpp][k][l][m].B[1];
	  cell[tpp][k][l][m].B[2]=d*cell[tpp][k][l][m].B[2];
	  
	  for(i=1;i<5;i++){
	    for(p=0;p<3;p++){
	      for(j=0;j<2;j++){
		cell[tpp][k][l][m].g[i][p][j]=((cell[tpp][k][l][m].E[0]*vec.e[i][p][j][0]+cell[tpp][k][l][m].E[1]*vec.e[i][p][j][1]+cell[tpp][k][l][m].E[2]*vec.e[i][p][j][2])/4.+
					       (cell[tpp][k][l][m].B[0]*vec.b[i][p][j][0]+cell[tpp][k][l][m].B[1]*vec.b[i][p][j][1]+cell[tpp][k][l][m].B[2]*vec.b[i][p][j][2])/8.);

	      }
	    }
	  }
	  }*/
      }
    }
  }

  return(0);
}
