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

cc 3d_pic397_domain.c -o 3d_pic397_domain -I/home/b/b31073/local/ta/include -L/home/b/b31073/local/ta/lib -lm -lgsl -lgslcblas -O3 -openmp -no-prec-div -no-prec-sqrt -xHost -static-intel -ipo -opt-mem-bandwidth2 -opt-calloc -unroll-aggressive -ansi-alias -opt-subscript-in-range -ltcmalloc

cc 3d_pic298_domain.c -o 3d_pic298_domain -I/home/b/b31073/local/ta_cray/include -L/home/b/b31073/local/ta_cray/lib -lm -lgsl -lgslcblas -O3 -h omp -h cache3 -h vector3 -h scalar3 -h ipa4 -h fp3

mpiicc 3d_pic307_domain.c -o 3d_pic307_domain -I/home/b/b31073/local/tb/include -L/home/b/b31073/local/tb/lib -lm -lgsl -lgslcblas -O3 -openmp -no-prec-div -no-prec-sqrt -xAVX -static-intel -ipo -opt-mem-bandwidth2 -opt-calloc -unroll-aggressive -ansi-alias -opt-subscript-in-range -ltcmalloc

without omp

mpifccpx 3d_pic458_domain.c -o 3d_pic458_domain -I/home/hp120084/k00175/local/include -L/home/hp120084/k00175/local/lib -lgsl -lm -lgslcblas -Kfast -Kpreex -Karray_private -Kocl -Ksimd=2 -Nsrc -Koptmsg=2 >a.txt 2>b2.txt 

with omp

mpifccpx 3d_pic356_domain.c -o 3d_pic356_domain -I/home/hp120084/k00175/local/include -L/home/hp120084/k00175/local/lib -lgsl -lm -lgslcblas -Kfast -Kpreex -Karray_private -Kocl -Ksimd=2 -Nsrc -Kopenmp -Koptmsg=2 >a.txt 2>b2.txt 

fccjx 3d_pic145_domain.c -o 3d_pic145_domain -Umpi -Uomp -I /home/y/y368/local/include -L /home/y/y368/local/lib/ -lm -lgsl -lgslcblas -O5 -Kfast -Kfsimple -KSPARC64VII -KV9 -KFMADD -Klib -Kprefetch -Kreduction

fccjxflat 3d_pic150_domain.c -o 3d_pic150_domain -Umpi -I /home/y/y368/local/include -L /home/y/y368/local/lib/ -lm -lgsl -lgslcblas -O5 -Kfast -Kfsimple -KSPARC64VII -KV9 -KFMADD -Klib -Kprefetch_model=FX1 -Kreduction
******************************************************************/
#ifdef _OPENMP
#include <omp.h>
#endif

#define version 458

#define q GSL_CONST_MKSA_ELECTRON_CHARGE
#define mi GSL_CONST_MKSA_MASS_PROTON
#define me (mi/100.)//GSL_CONST_MKSA_MASS_ELECTRON
#define mu0 (GSL_CONST_MKSA_VACUUM_PERMEABILITY)
#define e0 (GSL_CONST_MKSA_VACUUM_PERMITTIVITY)
#define kb GSL_CONST_MKSA_BOLTZMANN
#define Pi M_PI
#define C GSL_CONST_MKSA_SPEED_OF_LIGHT

#define dx 1.5//5.
#define dt 1.5E-9//5.E-9
#define Grid_Nx 4 //grid number x-axis
#define Grid_Ny 4 //grid number y-axis
#define Grid_Nz 512 //grid number z-axis

#define Np 16
#define V 1.2E7  //sun wind velocity  z-axis
#define Nall 5E10 //sun wind particle density
#define Ti 1e9  //1eV=1e4
#define Te 1e9

#define R 2.//=4.*dx
#define I 6.8e7 //6.8e10
#define alpha (Pi/2.)

#define Rs 1.
#define alphas Pi/2.

#define omega_y 0.//(-2.*Pi*10000.)
#define omega_z 0.

#define N0i (Nall/Np)
#define N0e (Nall/Np)

#define N1i (10.*Nall/Np)
#define N1e (10.*Nall/Np)

#define IMF_x 0.//3.5E-8
#define IMF_y 0.//(-3.125e-5) //-1.0E-7
#define IMF_z 0.//(-3.125e-5)//0.//3.5E-8

#define Step 300001

#define Absorb_grid 32  //field
#define Absorb_grid2 40  //particle
#define Absorb_grid3 40  //particle

#define THREAD_N 4
#define PROCESS_Nx 128
#define PROCESS_Ny 128

#define PACK_N 80000

#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr

#define PHI(x) 
#define DIPOLE(x) 

int active_thread=1;

/*******************************************************************
type holding grid infomation
******************************************************************/
typedef struct{
  double bx,by,bz;//total magnetic field   b0-b=induced magnetic field
  double b0x,b0y,b0z;//initial magnetic field
  double ex,ey,ez;//electric field
  double jix0,jiy0,jiz0;//ion current density
  double jex0,jey0,jez0;//electron current density
  double ni,ne;//number density
  double phi;
}Grid;

typedef struct{
  double jix0,jiy0,jiz0;//ion current density
  double jex0,jey0,jez0;//electron current density
  double ni,ne;//number density
}Grid_thread;

/*******************************************************************
type holding particle infomation
******************************************************************/
typedef struct particle_info{
  double x,y,z;
  double vx,vy,vz;
  double n;
  int flag;
  struct particle_info *prev_particle;
  struct particle_info *next_particle;
}Particle;

typedef struct{
  double k,l,m;
}klm;

Grid grid[Grid_Nx+4][Grid_Ny+4][Grid_Nz+4];
double sigma[2*(Grid_Nx+4)][2*(Grid_Ny+4)][2*(Grid_Nz+4)];
Grid_thread grid_thread[THREAD_N][Grid_Nx+4][Grid_Ny+4][Grid_Nz+4];
double Fave[27]; 

Particle *sort_particle[THREAD_N][Grid_Nx][Grid_Ny][Grid_Nz];
Particle *sort_particle_start[THREAD_N][Grid_Nx][Grid_Ny][Grid_Nz];

gsl_rng *rnd_p[THREAD_N];
gsl_rng *rnd_v[THREAD_N];
gsl_rng *rnd_i[THREAD_N];
unsigned int particle_flag[THREAD_N];

double ***rho_all;
double ***rho_all2;
double ***rho_all3;

const double dx3=dx*dx*dx;

int continue_flag=0;

double integrate_bx(double k,double l,double m);
double integrate_by(double k,double l,double m);
double integrate_bz(double k,double l,double m);
double fbx(double w,void *params);
double fby(double w,void *params);
double fbz(double w,void *params);

gsl_vector *normal_gaussian_vector(const gsl_rng *r,const size_t ndim);
gsl_matrix *covariance_matrix(const size_t ndim,const double *sigma0,const double *rho);
void cholesky_decomp_ltri(gsl_matrix *cov);
double *ran_multivar_gaussian(const gsl_rng *rng,const size_t ndim,const double *mu,const double *sigma0,const double *rho);

int cal_track2(Particle *p,const double m,const int myrank_x,const int myrank_y,const int c);
int cal_track3(Particle *p,const double m,const int myrank_x,const int myrank_y);
int cal_track4(Particle *p,const double m,const int myrank_x,const int myrank_y);
int del_particle_i(Particle *ion[],const int i,Particle *ion_start[]);
int del_particle_e(Particle *electron[],const int i,Particle *electron_start[]);
int shape_func(const Particle *p,double *ex,double *ey,double *ez,double *bx,double *by,double *bz);
int shape_func_v2(const Particle *p,double *ex,double *ey,double *ez,double *bx,double *by,double *bz);
int shape_func_ion0_n_2(const Particle *p,const int thread);
int shape_func_ion0_j_2(const Particle *p,const int thread);
int shape_func_electron0_n_2(const Particle *p,const int thread);
int shape_func_electron0_j_2(const Particle *p,const int thread);

int main(int argc,char *argv[])
{
  int myid,myrank,p;
  int myrank_x,myrank_y;
  double b0;

  MPI_Init(&argc,&argv);

  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&p);

  myrank=myid;

  myrank_x=myrank%PROCESS_Nx;
  myrank_y=myrank/PROCESS_Nx;

  if(p!=PROCESS_Nx*PROCESS_Ny){
    fprintf(stderr,"Process number is falut\n");
    exit(0);
  }

#ifdef _OPENMP
  omp_set_num_threads(active_thread);
  //omp_set_schedule(omp_sched_static,2);
#endif

  if(myrank_y*PROCESS_Nx+myrank_x==0){
    if(1.414*C*dt>dx){
      fprintf(stderr,"Courant error\n");
      //exit(-1);
    }else{
      fprintf(stderr,"dx/dt/C/1.414 = %f >1\n",dx/dt/C/1.414);
    }
	
    fprintf(stderr,"%d L=%E Debye=%f\n",version,pow(mu0*pow(Pi*R*R*I,2)/(8*Pi*Pi*Nall*mi*V*V),1./6.),sqrt(e0*kb*Te/Nall/q/q));
    fprintf(stderr,"L=%E\n",pow(mu0*pow(Pi*Pi*R*R*R*I,2)/(8*Pi*Pi*Nall*mi*V*V),1./6.));
    fprintf(stderr,"B@mp=%E\n",b0=sqrt(2*mu0*Nall*mi*V*V));
    fprintf(stderr,"wep=%E wec=%E\n",sqrt(Nall*q*q/e0/me),q/me*b0);
    fprintf(stderr,"wip=%E wic=%E\n",sqrt(Nall*q*q/e0/mi),q/mi*b0);
    fprintf(stderr,"tec=%E tic=%E\n",2.*Pi*me/q/b0,2.*Pi*mi/q/b0);
    fprintf(stderr,"tep=%E tip=%E\n",2.*Pi/sqrt(Nall*q*q/e0/me),2.*Pi/sqrt(Nall*q*q/e0/mi));
    fprintf(stderr,"vi_th=%E ve_th=%E\n",sqrt(2.*kb*Ti/mi),sqrt(2.*kb*Te/me));
    fprintf(stderr,"vs=%E\n",sqrt(2.*kb*Ti/mi+kb*Te/mi));
    fprintf(stderr,"Required steps %d\n",(int)(dx*Grid_Nz/V/dt));
  }


  if(argc>1){
    continue_flag=atoi(argv[1]);
  }else{   
    continue_flag=0;
  }

  if(myrank_x==0&&myrank_y==0){
    system("rm -rf finish");
  }

  main_func(myrank_x,myrank_y);

  MPI_Finalize();

  return(0);
}

int main_func(const int myrank_x,const int myrank_y)
{
  Particle *ion[THREAD_N];
  Particle *ion_start[THREAD_N];
  Particle *electron[THREAD_N];
  Particle *electron_start[THREAD_N];
  FILE *fp;
  char filename[256];

  fprintf(stderr,"start in (%d, %d)/(%d, %d)\n",myrank_x,myrank_y,PROCESS_Nx,PROCESS_Ny);

  //init_grid(myrank_x,myrank_y);

  //init_particle_flat(ion,ion_start,electron,electron_start,myrank_x,myrank_y);

  sprintf(filename,"suspend-%d-field.dat",myrank_x+myrank_y*PROCESS_Nx);
  if((fp=fopen(filename,"rb"))==NULL){
    init_grid(myrank_x,myrank_y);

    init_particle_flat(ion,ion_start,electron,electron_start,myrank_x,myrank_y);

    fprintf(stderr,"start_job\n");
    continue_flag=0;
  }else{
    fclose(fp);
    
    continue_job(ion,ion_start,electron,electron_start,myrank_x,myrank_y);
    fprintf(stderr,"continue_job %d\n",continue_flag);
  }//*/
  
  fdtd(ion,ion_start,electron,electron_start,myrank_x,myrank_y,continue_flag);

  return(0);
}

int init_grid(const int myrank_x,const int myrank_y)
{
  int k,l,m;
  const int absorb_n=4;
  const double a=1e-6;
  double x,y,z,r;

#pragma omp parallel for private(l,m)
  for(k=0;k<Grid_Nx+4;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][l][m].bx=0.;
	grid[k][l][m].by=0.;
	grid[k][l][m].bz=0.;
	
	grid[k][l][m].b0x=0.;
	grid[k][l][m].b0y=0.;
	grid[k][l][m].b0z=0.;
	
	grid[k][l][m].ex=0.;
	grid[k][l][m].ey=0.;
	grid[k][l][m].ez=0.;
	
	grid[k][l][m].jix0=0.;
	grid[k][l][m].jiy0=0.;
	grid[k][l][m].jiz0=0.;
	
	grid[k][l][m].jex0=0.;
	grid[k][l][m].jey0=0.;
	grid[k][l][m].jez0=0.;
	
	grid[k][l][m].ni=0.;
	grid[k][l][m].ne=0.;

	grid[k][l][m].phi=0;
      }
    }
  }

  external_current7(myrank_x,myrank_y,0);

#pragma omp parallel for private(l,m,x,y,z)
  for(k=0;k<2*(Grid_Nx+4);k++){
    for(l=0;l<2*(Grid_Ny+4);l++){
      for(m=0;m<2*(Grid_Nz+4);m++){
	sigma[k][l][m]=0.;

	if(k+2*myrank_x*Grid_Nx-4<=2*Absorb_grid){
	  x=(2*Absorb_grid-(k+2*myrank_x*Grid_Nx-4))/4.;
	}else if(k+2*myrank_x*Grid_Nx-4>=2*PROCESS_Nx*Grid_Nx-2*Absorb_grid){
	  x=(k+2*myrank_x*Grid_Nx-4-(2*PROCESS_Nx*Grid_Nx-2*Absorb_grid))/4.;
	}else{
	  x=0.;
	}

	if(l+2*myrank_y*Grid_Ny-4<=2*Absorb_grid){
	  y=(2*Absorb_grid-(l+2*myrank_y*Grid_Ny-4))/4.;
	}else if(l+2*myrank_y*Grid_Ny-4>=2*PROCESS_Ny*Grid_Ny-2*Absorb_grid){
	  y=(l+2*myrank_y*Grid_Ny-4-(2*PROCESS_Ny*Grid_Ny-2*Absorb_grid))/4.;
	}else{
	  y=0.;
	}

	if(m-4<=2*Absorb_grid){
	  z=(2*Absorb_grid-(m-4))/4.;
	}else if(m-4>=2*Grid_Nz-2*Absorb_grid){
	  z=(m-4-(2*Grid_Nz-2*Absorb_grid))/4.;
	}else{
	  z=0.;
	}

	sigma[k][l][m]=a*(0.1*sqrt(pow(x,4)+pow(y,4)+pow(z,4))
			  +0.01*sqrt(pow(x,12)+pow(y,12)+pow(z,12)));
      }
    }
  }

  PHI(
      rho_all=malloc(sizeof(double**)*Grid_Nx*PROCESS_Nx);
      
      for(k=0;k<Grid_Nx*PROCESS_Nx;k++){
	rho_all[k]=malloc(sizeof(double*)*Grid_Ny*PROCESS_Ny);
	for(l=0;l<Grid_Ny*PROCESS_Ny;l++){
	  rho_all[k][l]=malloc(sizeof(double)*Grid_Nz);
	  if(rho_all[k][l]==NULL){
	    printf("Don't get MEMORY\n");
	    exit(-1);
	  }
	}
      }
      
      rho_all2=malloc(sizeof(double**)*Grid_Nz);
      
      for(k=0;k<Grid_Nz;k++){
	rho_all2[k]=malloc(sizeof(double*)*Grid_Nx*PROCESS_Nx);
	for(l=0;l<Grid_Nx*PROCESS_Nx;l++){
	  rho_all2[k][l]=malloc(sizeof(double)*Grid_Ny*PROCESS_Ny);
	  if(rho_all2[k][l]==NULL){
	    printf("Don't get MEMORY\n");
	    exit(-1);
	  }
	}
      }
      
      rho_all3=malloc(sizeof(double**)*Grid_Ny*PROCESS_Ny);
      
      for(k=0;k<Grid_Ny*PROCESS_Ny;k++){
	rho_all3[k]=malloc(sizeof(double*)*Grid_Nz);
	for(l=0;l<Grid_Nz;l++){
	  rho_all3[k][l]=malloc(sizeof(double)*Grid_Nx*PROCESS_Nx);
	  if(rho_all3[k][l]==NULL){
	    printf("Don't get MEMORY\n");
	    exit(-1);
	  }
	}
      }
      );

  return(0);
}

int init_particle_flat(Particle *ion[],Particle *ion_start[],Particle *electron[],Particle *electron_start[],const int myrank_x,const int myrank_y)
{
  int i,j;
  int k,l,m,n;
  const gsl_rng_type *T;
  const int loop=(Np*Grid_Nx*Grid_Ny*Grid_Nz)/THREAD_N;
  double mu[3];
  double sigma0[3];
  double rho[3];
  double *vt;

  mu[0]=0.;
  mu[1]=0.;
  mu[2]=V;

  sigma0[0]=sqrt(kb*Ti/mi);
  sigma0[1]=sqrt(kb*Ti/mi);
  sigma0[2]=sqrt(kb*Ti/mi);

  rho[0]=0.;
  rho[1]=0.;
  rho[2]=0.;

  //T=gsl_rng_ranlxd2;
  T=gsl_rng_rand;

  for(i=0;i<THREAD_N;i++){
    rnd_p[i]=gsl_rng_alloc(T);
    rnd_v[i]=gsl_rng_alloc(T);
    rnd_i[i]=gsl_rng_alloc(T);
    gsl_rng_set(rnd_p[i],(myrank_y*PROCESS_Nx+myrank_x)*10000+i*100);
    gsl_rng_set(rnd_v[i],(myrank_y*PROCESS_Nx+myrank_x)*10000+i*100+50);
    gsl_rng_set(rnd_i[i],i*10);
    particle_flag[i]=1;
  }

  for(i=0;i<THREAD_N;i++){
    ion[i]=ion_start[i]=NULL;
    electron[i]=electron_start[i]=NULL;
  }

#pragma omp parallel for private(j)
  for(i=0;i<THREAD_N;i++){
    if((ion[i]=malloc(sizeof(Particle)))!=NULL){
      ion_start[i]=ion[i];
      ion[i]->prev_particle=NULL;
      ion[i]->next_particle=NULL;
    }

    for(j=0;j<loop;j++){
      if((ion[i]->next_particle=(Particle*)malloc(sizeof(Particle)))!=NULL){
	ion[i]->next_particle->prev_particle=ion[i];
	ion[i]=ion[i]->next_particle;
	ion[i]->next_particle=NULL;
      }else{
	exit(0);
      }
    }
  }

#pragma omp parallel for private(j)
  for(i=0;i<THREAD_N;i++){
    if((electron[i]=malloc(sizeof(Particle)))!=NULL){
      electron_start[i]=electron[i];
      electron[i]->prev_particle=NULL;
      electron[i]->next_particle=NULL;
    }

    for(j=0;j<loop;j++){
      if((electron[i]->next_particle=(Particle*)malloc(sizeof(Particle)))!=NULL){
	electron[i]->next_particle->prev_particle=electron[i];
	electron[i]=electron[i]->next_particle;
	electron[i]->next_particle=NULL;
      }else{
	exit(0);
      }
    }
  }

#pragma omp parallel for private(k,l,m,n) 
  for(i=0;i<THREAD_N;i++){
    k=l=m=n=0;
    ion[i]=ion_start[i];
    k=i*Grid_Nx/THREAD_N;

    while(ion[i]->next_particle!=NULL){
      ion[i]->x=gsl_ran_flat(rnd_p[i],2.*dx+k*dx,2.*dx+(k+1)*dx);
      ion[i]->y=gsl_ran_flat(rnd_p[i],2.*dx+l*dx,2.*dx+(l+1)*dx);
      ion[i]->z=gsl_ran_flat(rnd_p[i],2.*dx+m*dx,2.*dx+(m+1)*dx);

      vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
      ion[i]->vx=vt[0];
      ion[i]->vy=vt[1];
      ion[i]->vz=vt[2];
      free(vt);

      ion[i]->n=N0i;
      //ion[i]->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
      ion[i]->flag=1;
      particle_flag[i]++;

      n++;
      if(n==Np){
	n=0;
	m++;
      }

      if(m==Grid_Nz){
	m=0;
	l++;
      }

      if(l==Grid_Ny){
	l=0;
	k++;
      }

      ion[i]=ion[i]->next_particle;
    }

    ion[i]->x=gsl_ran_flat(rnd_p[i],2.*dx+k*dx,2.*dx+(k+1)*dx);
    ion[i]->y=gsl_ran_flat(rnd_p[i],2.*dx+l*dx,2.*dx+(l+1)*dx);
    ion[i]->z=gsl_ran_flat(rnd_p[i],2.*dx+m*dx,2.*dx+(m+1)*dx);
      
    vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
    ion[i]->vx=vt[0];
    ion[i]->vy=vt[1];
    ion[i]->vz=vt[2];
    free(vt);

    ion[i]->n=N0i;
    //ion[i]->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
    ion[i]->flag=1;
    particle_flag[i]++;
  }

  sigma0[0]=sqrt(kb*Te/me);
  sigma0[1]=sqrt(kb*Te/me);
  sigma0[2]=sqrt(kb*Te/me);

#pragma omp parallel for private(k,l,m,n) 
  for(i=0;i<THREAD_N;i++){
    k=l=m=n=0;
    electron[i]=electron_start[i];
    k=i*Grid_Nx/THREAD_N;

    while(electron[i]->next_particle!=NULL){
      electron[i]->x=gsl_ran_flat(rnd_p[i],2.*dx+k*dx,2.*dx+(k+1)*dx);
      electron[i]->y=gsl_ran_flat(rnd_p[i],2.*dx+l*dx,2.*dx+(l+1)*dx);
      electron[i]->z=gsl_ran_flat(rnd_p[i],2.*dx+m*dx,2.*dx+(m+1)*dx);

      vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
      electron[i]->vx=vt[0];
      electron[i]->vy=vt[1];
      electron[i]->vz=vt[2];
      free(vt);

      electron[i]->n=N0e;
      //electron[i]->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
      electron[i]->flag=1;
      particle_flag[i]++;

      n++;
      if(n==Np){
	n=0;
	m++;
      }

      if(m==Grid_Nz){
	m=0;
	l++;
      }

      if(l==Grid_Ny){
	l=0;
	k++;
      }

      electron[i]=electron[i]->next_particle;
    }

    electron[i]->x=gsl_ran_flat(rnd_p[i],2.*dx+k*dx,2.*dx+(k+1)*dx);
    electron[i]->y=gsl_ran_flat(rnd_p[i],2.*dx+l*dx,2.*dx+(l+1)*dx);
    electron[i]->z=gsl_ran_flat(rnd_p[i],2.*dx+m*dx,2.*dx+(m+1)*dx);
    
    vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
    electron[i]->vx=vt[0];
    electron[i]->vy=vt[1];
    electron[i]->vz=vt[2];
    free(vt);

    electron[i]->n=N0e;
    //electron[i]->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
    electron[i]->flag=1;
    particle_flag[i]++;
  }

#pragma omp parallel for private(k,l,m)
  for(i=0;i<THREAD_N;i++){
    for(k=0;k<Grid_Nx;k++){
      for(l=0;l<Grid_Ny;l++){
	for(m=0;m<Grid_Nz;m++){
	  if((sort_particle[i][k][l][m]=(Particle*)malloc(sizeof(Particle)))!=NULL){
	    sort_particle_start[i][k][l][m]=sort_particle[i][k][l][m];
	    sort_particle[i][k][l][m]->prev_particle=NULL;
	    sort_particle[i][k][l][m]->next_particle=NULL;
	  }
	}
      }
    }
  }

 return(0);
}

int fdtd(Particle *ion[],Particle *ion_start[],Particle *electron[],Particle *electron_start[],const int myrank_x,const int myrank_y,const int c_loop)
{
  int i,j,c;
  int k,l,m;
  int c_output;
  int c_thrust;
  int c_inject;
  int c_add;
  int c_srp;
  int c_suspend;
  time_t time1,time2;
  FILE *fp_m,*fp1,*fp_w;
  char filename[256];
  double M_plasma;

  const double ax0=Absorb_grid2*dx-myrank_x*Grid_Nx*dx;
  const double ay0=Absorb_grid2*dx-myrank_y*Grid_Ny*dx;
  const double az0=Absorb_grid2*dx;
  const double ax1=(PROCESS_Nx*Grid_Nx+4-Absorb_grid2)*dx-myrank_x*Grid_Nx*dx;
  const double ay1=(PROCESS_Ny*Grid_Ny+4-Absorb_grid2)*dx-myrank_y*Grid_Ny*dx;
  const double az1=(Grid_Nz+4-Absorb_grid2)*dx;
  /*const double d3x=(Grid_Nx+3)*dx;
  const double d3y=(Grid_Ny+3)*dx;
  const double d3z=(Grid_Nz+3)*dx;
  const double dx0=dx;
  const double dy0=dx;
  const double dz0=dx;*/
  const double d3x=(Grid_Nx+4)*dx;
  const double d3y=(Grid_Ny+4)*dx;
  const double d3z=(Grid_Nz+4)*dx;
  const double dx0=0.;
  const double dy0=0.;
  const double dz0=0.;

  const double dth=dt*0.5;
  const double dxd=2.*dx;

  time(&time1);

  c_output=0;
  c_thrust=0;
  c_inject=0;
  c_add=0;
  c_srp=0;
  c_suspend=0;

  add_ion_flat(ion,ion_start,myrank_x,myrank_y);
  add_electron_flat(electron,electron_start,myrank_x,myrank_y);

  PHI(
      renew_phi3(myrank_x,myrank_y);
      
      sr_phi(myrank_x,myrank_y);
      
      sr_e(myrank_x,myrank_y);
      );

  time(&time2);
  fprintf(stderr,"%d %f\n",c_loop,difftime(time2,time1));

  for(c=c_loop;c<Step+c_loop;c++){
    renew_grid();

    if(c_output>4999){//999
      c_output=0;
    }

    if(c_thrust>499){//999
      c_thrust=0;
      sorting(ion,ion_start);
      sorting(electron,electron_start);
    }

    if(c_add>4){//////////4
      add_ion_flat(ion,ion_start,myrank_x,myrank_y);
      add_electron_flat(electron,electron_start,myrank_x,myrank_y);
      c_add=0;
      time(&time2);
      fprintf(stderr,"%d %f\n",c,difftime(time2,time1));
    }
    // fprintf(stderr,"%d\n",c);


    if(c>4999&&c_inject>99){
      fprintf(stderr,"%d \n",c);
      //inject_ion_para4(ion,ion_start,myrank_x,myrank_y);
      //inject_electron_para4(electron,electron_start,myrank_x,myrank_y);
      //inject_ion_perp4(ion,ion_start,myrank_x,myrank_y);
      //inject_electron_perp4(electron,electron_start,myrank_x,myrank_y);
      c_inject=0;
    }

    if(c_srp>0){
      c_srp=0;
    }

    if(c_suspend>999){
      c_suspend=0;
    }

    renew_b4_2();

    sr_b(myrank_x,myrank_y);
  
#pragma omp parallel for num_threads(active_thread)
    for(i=0;i<THREAD_N;i++){
      ion[i]=ion_start[i];
      while(ion[i]->next_particle!=NULL){
	if(ion[i]->x<ax0||ion[i]->x>ax1||
	   ion[i]->y<ay0||ion[i]->y>ay1||
	   ion[i]->z<az0||ion[i]->z>az1){
	  ion[i]->x+=ion[i]->vx*dth;
	  ion[i]->y+=ion[i]->vy*dth;
	  ion[i]->z+=ion[i]->vz*dth;
	}else{
	  //cal_track(ion[i],mi);
	  //cal_track2(ion[i],mi,myrank_x,myrank_y,c);
	  cal_track4(ion[i],mi,myrank_x,myrank_y);
	}

	if(ion[i]->x<d3x&&ion[i]->x>=dx0&&
	   ion[i]->y<d3y&&ion[i]->y>=dx0&&
	   ion[i]->z<d3z&&ion[i]->z>=dx0){
	  shape_func_ion0_j_2(ion[i],i);
	}
	ion[i]=ion[i]->next_particle;
      }

      if(ion[i]->x<ax0||ion[i]->x>ax1||
	 ion[i]->y<ay0||ion[i]->y>ay1||
	 ion[i]->z<az0||ion[i]->z>az1){
	ion[i]->x+=ion[i]->vx*dth;
	ion[i]->y+=ion[i]->vy*dth;
	ion[i]->z+=ion[i]->vz*dth;
      }else{
	//cal_track(ion[i],mi);
	//cal_track2(ion[i],mi,myrank_x,myrank_y,c);
	cal_track4(ion[i],mi,myrank_x,myrank_y);
      }
      
      if(ion[i]->x<d3x&&ion[i]->x>=dx0&&
	 ion[i]->y<d3y&&ion[i]->y>=dx0&&
	 ion[i]->z<d3z&&ion[i]->z>=dx0){
	shape_func_ion0_j_2(ion[i],i);
      }
      
      electron[i]=electron_start[i];
      while(electron[i]->next_particle!=NULL){	
	if(electron[i]->x<ax0||electron[i]->x>ax1||
	   electron[i]->y<ay0||electron[i]->y>ay1||
	   electron[i]->z<az0||electron[i]->z>az1){
	  electron[i]->x+=electron[i]->vx*dth;
	  electron[i]->y+=electron[i]->vy*dth;
	  electron[i]->z+=electron[i]->vz*dth;
	}else{
	  //cal_track(electron[i],-me);
	  //cal_track2(electron[i],-me,myrank_x,myrank_y,c);
	  cal_track4(electron[i],-me,myrank_x,myrank_y);
	}

	if(electron[i]->x<d3x&&electron[i]->x>=dx0&&
	   electron[i]->y<d3y&&electron[i]->y>=dx0&&
	   electron[i]->z<d3z&&electron[i]->z>=dx0){
	  shape_func_electron0_j_2(electron[i],i);
	}
	electron[i]=electron[i]->next_particle;
      }

      if(electron[i]->x<ax0||electron[i]->x>ax1||
	 electron[i]->y<ay0||electron[i]->y>ay1||
	 electron[i]->z<az0||electron[i]->z>az1){
	electron[i]->x+=electron[i]->vx*dth;
	electron[i]->y+=electron[i]->vy*dth;
	electron[i]->z+=electron[i]->vz*dth;
      }else{
	//cal_track(electron[i],-me);
	//cal_track2(electron[i],-me,myrank_x,myrank_y,c);
	cal_track4(electron[i],-me,myrank_x,myrank_y);
      }
      
      if(electron[i]->x<d3x&&electron[i]->x>=dx0&&
	 electron[i]->y<d3y&&electron[i]->y>=dx0&&
	 electron[i]->z<d3z&&electron[i]->z>=dx0){
	shape_func_electron0_j_2(electron[i],i);
      }
    }

    renew_b4_2();

    sr_b(myrank_x,myrank_y);
       
#pragma omp parallel for num_threads(active_thread)
    for(i=0;i<THREAD_N;i++){
      ion[i]=ion_start[i];
      while(ion[i]->next_particle!=NULL){
	ion[i]->x+=ion[i]->vx*dth;
	ion[i]->y+=ion[i]->vy*dth;
	ion[i]->z+=ion[i]->vz*dth;
	
	if(ion[i]->x<d3x&&ion[i]->x>=dx0&&
	   ion[i]->y<d3y&&ion[i]->y>=dx0&&
	   ion[i]->z<d3z&&ion[i]->z>=dx0){
	  shape_func_ion0_n_2(ion[i],i);
	}
	ion[i]=ion[i]->next_particle;
      }

      ion[i]->x+=ion[i]->vx*dth;
      ion[i]->y+=ion[i]->vy*dth;
      ion[i]->z+=ion[i]->vz*dth;
      
      if(ion[i]->x<d3x&&ion[i]->x>=dx0&&
	 ion[i]->y<d3y&&ion[i]->y>=dx0&&
	 ion[i]->z<d3z&&ion[i]->z>=dx0){
	shape_func_ion0_n_2(ion[i],i);
      }
  
      electron[i]=electron_start[i];
      while(electron[i]->next_particle!=NULL){	
	electron[i]->x+=electron[i]->vx*dth;
	electron[i]->y+=electron[i]->vy*dth;
	electron[i]->z+=electron[i]->vz*dth;
	
	if(electron[i]->x<d3x&&electron[i]->x>=dx0&&
	   electron[i]->y<d3y&&electron[i]->y>=dx0&&
	   electron[i]->z<d3z&&electron[i]->z>=dx0){
	  shape_func_electron0_n_2(electron[i],i);
	}
	electron[i]=electron[i]->next_particle;
      }
      
      electron[i]->x+=electron[i]->vx*dth;
      electron[i]->y+=electron[i]->vy*dth;
      electron[i]->z+=electron[i]->vz*dth;
      
      if(electron[i]->x<d3x&&electron[i]->x>=dx0&&
	 electron[i]->y<d3y&&electron[i]->y>=dx0&&
	 electron[i]->z<d3z&&electron[i]->z>=dx0){
	shape_func_electron0_n_2(electron[i],i);
      }
    }
    
    for(i=0;i<THREAD_N;i++){
#pragma omp parallel for private(l,m)
      for(k=0;k<Grid_Nx+4;k++){
	for(l=0;l<Grid_Ny+4;l++){
	  for(m=0;m<Grid_Nz+4;m++){
	    grid[k][l][m].jix0+=grid_thread[i][k][l][m].jix0;
	    grid[k][l][m].jiy0+=grid_thread[i][k][l][m].jiy0;
	    grid[k][l][m].jiz0+=grid_thread[i][k][l][m].jiz0;
	    grid[k][l][m].jex0+=grid_thread[i][k][l][m].jex0;
	    grid[k][l][m].jey0+=grid_thread[i][k][l][m].jey0;
	    grid[k][l][m].jez0+=grid_thread[i][k][l][m].jez0;

	    grid[k][l][m].ni+=grid_thread[i][k][l][m].ni;
	    grid[k][l][m].ne+=grid_thread[i][k][l][m].ne;
	  }
	}
      }
    }

    //external_current(myrank_x,myrank_y,c);
    //external_current3(myrank_x,myrank_y,c);
    //external_current6(myrank_x,myrank_y,c);

    //sr_current(myrank_x,myrank_y);
    sr_current2(myrank_x,myrank_y);

    renew_e4_2();//use j

    //conductor(ion,ion_start,electron,electron_start,myrank_x,myrank_y);

    sr_e(myrank_x,myrank_y);

    PHI(
	renew_phi3(myrank_x,myrank_y);
	
	sr_phi(myrank_x,myrank_y);
	
	sr_e(myrank_x,myrank_y);
	);

    if(c_srp==0){
      sr_particle(ion,ion_start,electron,electron_start,myrank_x,myrank_y);
    }

    if(c_output==0){
       output(myrank_x,myrank_y,c);
    }

    if(c_thrust==0){
      thrust_f(myrank_x,myrank_y,c,ion,ion_start,electron,electron_start);

      if(myrank_x==0&&myrank_y==0){
	if(fopen("finish","r")!=NULL){
	  exit(0);
	}
      }
    }
    thrust_fave(myrank_x,myrank_y,c,ion,ion_start,electron,electron_start);

    if(c_suspend==0){
      suspend_job(ion,ion_start,electron,electron_start,myrank_x,myrank_y);
      //all_particles(ion,ion_start,electron,electron_start,myrank_x,myrank_y,c);
    }

    /*if(myrank_x==32&&myrank_y==32){
      fp_w=fopen("all_wave32x32.txt","a");
      fprintf(fp_w,"%E %E %E %E %E %E %E %E %E %E %E %E %E %E %E %E %E %E %E %E %E %E %E %E\n",
	      grid[4][4][64].ex,grid[4][4][64].ey,grid[4][4][64].ez,grid[4][4][64].bx-grid[4][4][64].b0x,grid[4][4][64].by-grid[4][4][64].b0y,grid[4][4][64].bz-grid[4][4][64].b0z,
	      grid[4][4][96].ex,grid[4][4][96].ey,grid[4][4][96].ez,grid[4][4][96].bx-grid[4][4][96].b0x,grid[4][4][96].by-grid[4][4][96].b0y,grid[4][4][96].bz-grid[4][4][96].b0z,
	      grid[4][4][128].ex,grid[4][4][128].ey,grid[4][4][128].ez,grid[4][4][128].bx-grid[4][4][128].b0x,grid[4][4][128].by-grid[4][4][128].b0y,grid[4][4][128].bz-grid[4][4][128].b0z,
	      grid[4][4][150].ex,grid[4][4][150].ey,grid[4][4][150].ez,grid[4][4][150].bx-grid[4][4][150].b0x,grid[4][4][150].by-grid[4][4][150].b0y,grid[4][4][150].bz-grid[4][4][150].b0z);
      fclose(fp_w);
      }*/

    c_output++;
    c_thrust++;
    c_add++;
    c_inject++;
    c_srp++;
    c_suspend++;
  }

  return(0);
}

int conductor(Particle *ion[],Particle *ion_start[],Particle *electron[],Particle *electron_start[],const int myrank_x,const int myrank_y)
{
  int k,l,m;
  int i;

  for(k=0;k<Grid_Nx+4;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	if((pow(k-2+Grid_Nx*myrank_x-Grid_Nx*PROCESS_Nx/2.,2)+
	    pow(l-2+Grid_Ny*myrank_y-Grid_Ny*PROCESS_Ny/2.,2)+
	    pow(m-2-Grid_Nz/2.,2))<20.*20.){
	  grid[k][l][m].ex=0.;
	  grid[k][l][m].ey=0.;
	  grid[k][l][m].ez=0.;
	}
      }
    }
  }

  for(i=0;i<THREAD_N;i++){
    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){
      if((pow(ion[i]->x/dx+Grid_Nx*myrank_x-Grid_Nx*PROCESS_Nx/2.,2)+
	  pow(ion[i]->y/dx+Grid_Ny*myrank_y-Grid_Ny*PROCESS_Ny/2.,2)+
	  pow(ion[i]->z/dx-Grid_Nz/2.,2))<20.*20.){

	del_particle_i(ion,i,ion_start);
      }
      ion[i]=ion[i]->next_particle;
    }

    electron[i]=electron_start[i];
    while(electron[i]->next_particle!=NULL){
      if((pow(electron[i]->x/dx+Grid_Nx*myrank_x-Grid_Nx*PROCESS_Nx/2.,2)+
	  pow(electron[i]->y/dx+Grid_Ny*myrank_y-Grid_Ny*PROCESS_Ny/2.,2)+
	  pow(electron[i]->z/dx-Grid_Nz/2.,2))<20.*20.){

	del_particle_e(electron,i,electron_start);
      }
      electron[i]=electron[i]->next_particle;
    }

  }

  return(0);
}

int sr_b(const int myrank_x,const int myrank_y)
{  
  int src,dest,tag,count;
  MPI_Status stat;
  MPI_Request request;
  int i,j;
  int k,l,m;

  double send_buf2x[2*(Grid_Ny+4)*(Grid_Nz+4)*3];
  double send_buf2y[2*(Grid_Nx+4)*(Grid_Nz+4)*3];
  double recv_buf2x[2*(Grid_Ny+4)*(Grid_Nz+4)*3];
  double recv_buf2y[2*(Grid_Nx+4)*(Grid_Nz+4)*3];

  //x direction

  tag=1000;

  count=2*(Grid_Ny+4)*(Grid_Nz+4)*3;

  i=0;
  for(k=Grid_Nx;k<Grid_Nx+2;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	send_buf2x[i]=grid[k][l][m].bx;
	send_buf2x[i+1]=grid[k][l][m].by;
	send_buf2x[i+2]=grid[k][l][m].bz;
	i+=3;
      }
    }
  }

  src=myrank_y*PROCESS_Nx+myrank_x-1;

  if(myrank_x==0){
    src=myrank_y*PROCESS_Nx+PROCESS_Nx-1;
  }

  MPI_Irecv(&recv_buf2x,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=myrank_y*PROCESS_Nx+myrank_x+1;

  if(myrank_x==PROCESS_Nx-1){
    dest=myrank_y*PROCESS_Nx+0;
  }

  MPI_Send(&send_buf2x,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  for(k=0;k<2;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][l][m].bx=recv_buf2x[i];
	grid[k][l][m].by=recv_buf2x[i+1];
	grid[k][l][m].bz=recv_buf2x[i+2];
	i+=3;
      }
    }
  }

  i=0;
  for(k=2;k<4;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	send_buf2x[i]=grid[k][l][m].bx;
	send_buf2x[i+1]=grid[k][l][m].by;
	send_buf2x[i+2]=grid[k][l][m].bz;
	i+=3;
      }
    }
  }

  src=myrank_y*PROCESS_Nx+myrank_x+1;

  if(myrank_x==PROCESS_Nx-1){
    src=myrank_y*PROCESS_Nx+0;
  }

  MPI_Irecv(&recv_buf2x,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=myrank_y*PROCESS_Nx+myrank_x-1;

  if(myrank_x==0){
    dest=myrank_y*PROCESS_Nx+PROCESS_Nx-1;
  }

  MPI_Send(&send_buf2x,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  for(k=Grid_Nx+2;k<Grid_Nx+4;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][l][m].bx=recv_buf2x[i];
	grid[k][l][m].by=recv_buf2x[i+1];
	grid[k][l][m].bz=recv_buf2x[i+2];
	i+=3;
      }
    }
  }

  //y direction

  count=2*(Grid_Nx+4)*(Grid_Nz+4)*3;

  i=0;
  for(k=0;k<Grid_Nx+4;k++){
    for(l=Grid_Ny;l<Grid_Ny+2;l++){
      for(m=0;m<Grid_Nz+4;m++){
	send_buf2y[i]=grid[k][l][m].bx;
	send_buf2y[i+1]=grid[k][l][m].by;
	send_buf2y[i+2]=grid[k][l][m].bz;
	i+=3;
      }
    }
  }

  src=(myrank_y-1)*PROCESS_Nx+myrank_x;

  if(myrank_y==0){
    src=(PROCESS_Ny-1)*PROCESS_Nx+myrank_x;
  }

  MPI_Irecv(&recv_buf2y,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=(myrank_y+1)*PROCESS_Nx+myrank_x;

  if(myrank_y==PROCESS_Ny-1){
    dest=0*PROCESS_Nx+myrank_x;
  }

  MPI_Send(&send_buf2y,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  for(k=0;k<Grid_Nx+4;k++){
    for(l=0;l<2;l++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][l][m].bx=recv_buf2y[i];
	grid[k][l][m].by=recv_buf2y[i+1];
	grid[k][l][m].bz=recv_buf2y[i+2];
	i+=3;
      }
    }
  }

  i=0;
  for(k=0;k<Grid_Nx+4;k++){
    for(l=2;l<4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	send_buf2y[i]=grid[k][l][m].bx;
	send_buf2y[i+1]=grid[k][l][m].by;
	send_buf2y[i+2]=grid[k][l][m].bz;
	i+=3;
      }
    }
  }

  src=(myrank_y+1)*PROCESS_Nx+myrank_x;

  if(myrank_y==PROCESS_Ny-1){
    src=0*PROCESS_Nx+myrank_x;
  }

  MPI_Irecv(&recv_buf2y,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=(myrank_y-1)*PROCESS_Nx+myrank_x;

  if(myrank_y==0){
    dest=(PROCESS_Ny-1)*PROCESS_Nx+myrank_x;
  }

  MPI_Send(&send_buf2y,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  for(k=0;k<Grid_Nx+4;k++){
    for(l=Grid_Ny+2;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][l][m].bx=recv_buf2y[i];
	grid[k][l][m].by=recv_buf2y[i+1];
	grid[k][l][m].bz=recv_buf2y[i+2];
	i+=3;
      }
    }
  }

  return(0);
}

int sr_e(const int myrank_x,const int myrank_y)
{  
  int src,dest,tag,count;
  MPI_Status stat;
  MPI_Request request;
  int i,j;
  int k,l,m;

  double send_buf2x[2*(Grid_Ny+4)*(Grid_Nz+4)*3];
  double send_buf2y[2*(Grid_Nx+4)*(Grid_Nz+4)*3];
  double recv_buf2x[2*(Grid_Ny+4)*(Grid_Nz+4)*3];
  double recv_buf2y[2*(Grid_Nx+4)*(Grid_Nz+4)*3];

  //x direction

  tag=1000;

  count=2*(Grid_Ny+4)*(Grid_Nz+4)*3;

  i=0;
  for(k=Grid_Nx;k<Grid_Nx+2;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	send_buf2x[i]=grid[k][l][m].ex;
	send_buf2x[i+1]=grid[k][l][m].ey;
	send_buf2x[i+2]=grid[k][l][m].ez;
	i+=3;
      }
    }
  }

  src=myrank_y*PROCESS_Nx+myrank_x-1;

  if(myrank_x==0){
    src=myrank_y*PROCESS_Nx+PROCESS_Nx-1;
  }

  MPI_Irecv(&recv_buf2x,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=myrank_y*PROCESS_Nx+myrank_x+1;

  if(myrank_x==PROCESS_Nx-1){
    dest=myrank_y*PROCESS_Nx+0;
  }

  MPI_Send(&send_buf2x,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  for(k=0;k<2;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][l][m].ex=recv_buf2x[i];
	grid[k][l][m].ey=recv_buf2x[i+1];
	grid[k][l][m].ez=recv_buf2x[i+2];
	i+=3;
      }
    }
  }

  i=0;
  for(k=2;k<4;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	send_buf2x[i]=grid[k][l][m].ex;
	send_buf2x[i+1]=grid[k][l][m].ey;
	send_buf2x[i+2]=grid[k][l][m].ez;
	i+=3;
      }
    }
  }

  src=myrank_y*PROCESS_Nx+myrank_x+1;

  if(myrank_x==PROCESS_Nx-1){
    src=myrank_y*PROCESS_Nx+0;
  }

  MPI_Irecv(&recv_buf2x,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=myrank_y*PROCESS_Nx+myrank_x-1;

  if(myrank_x==0){
    dest=myrank_y*PROCESS_Nx+PROCESS_Nx-1;
  }

  MPI_Send(&send_buf2x,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  for(k=Grid_Nx+2;k<Grid_Nx+4;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][l][m].ex=recv_buf2x[i];
	grid[k][l][m].ey=recv_buf2x[i+1];
	grid[k][l][m].ez=recv_buf2x[i+2];
	i+=3;
      }
    }
  }

  //y direction

  count=2*(Grid_Nx+4)*(Grid_Nz+4)*3;

  i=0;
  for(k=0;k<Grid_Nx+4;k++){
    for(l=Grid_Ny;l<Grid_Ny+2;l++){
      for(m=0;m<Grid_Nz+4;m++){
	send_buf2y[i]=grid[k][l][m].ex;
	send_buf2y[i+1]=grid[k][l][m].ey;
	send_buf2y[i+2]=grid[k][l][m].ez;
	i+=3;
      }
    }
  }

  src=(myrank_y-1)*PROCESS_Nx+myrank_x;

  if(myrank_y==0){
    src=(PROCESS_Ny-1)*PROCESS_Nx+myrank_x;
  }

  MPI_Irecv(&recv_buf2y,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=(myrank_y+1)*PROCESS_Nx+myrank_x;

  if(myrank_y==PROCESS_Ny-1){
    dest=0*PROCESS_Nx+myrank_x;
  }

  MPI_Send(&send_buf2y,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  for(k=0;k<Grid_Nx+4;k++){
    for(l=0;l<2;l++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][l][m].ex=recv_buf2y[i];
	grid[k][l][m].ey=recv_buf2y[i+1];
	grid[k][l][m].ez=recv_buf2y[i+2];
	i+=3;
      }
    }
  }

  i=0;
  for(k=0;k<Grid_Nx+4;k++){
    for(l=2;l<4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	send_buf2y[i]=grid[k][l][m].ex;
	send_buf2y[i+1]=grid[k][l][m].ey;
	send_buf2y[i+2]=grid[k][l][m].ez;
	i+=3;
      }
    }
  }

  src=(myrank_y+1)*PROCESS_Nx+myrank_x;

  if(myrank_y==PROCESS_Ny-1){
    src=0*PROCESS_Nx+myrank_x;
  }

  MPI_Irecv(&recv_buf2y,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=(myrank_y-1)*PROCESS_Nx+myrank_x;

  if(myrank_y==0){
    dest=(PROCESS_Ny-1)*PROCESS_Nx+myrank_x;
  }

  MPI_Send(&send_buf2y,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  for(k=0;k<Grid_Nx+4;k++){
    for(l=Grid_Ny+2;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][l][m].ex=recv_buf2y[i];
	grid[k][l][m].ey=recv_buf2y[i+1];
	grid[k][l][m].ez=recv_buf2y[i+2];
	i+=3;
      }
    }
  }

  return(0);
}

int sr_phi(const int myrank_x,const int myrank_y)
{  
  int src,dest,tag,count;
  MPI_Status stat;
  MPI_Request request;
  int i,j;
  int k,l,m;

  double send_buf2x[2*(Grid_Ny+4)*(Grid_Nz+4)];
  double send_buf2y[2*(Grid_Nx+4)*(Grid_Nz+4)];
  double recv_buf2x[2*(Grid_Ny+4)*(Grid_Nz+4)];
  double recv_buf2y[2*(Grid_Nx+4)*(Grid_Nz+4)];

  //x direction

  tag=1000;

  count=2*(Grid_Ny+4)*(Grid_Nz+4);

  i=0;
  for(k=Grid_Nx;k<Grid_Nx+2;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	send_buf2x[i]=grid[k][l][m].phi;
	i+=1;
      }
    }
  }

  src=myrank_y*PROCESS_Nx+myrank_x-1;

  if(myrank_x==0){
    src=myrank_y*PROCESS_Nx+PROCESS_Nx-1;
  }

  MPI_Irecv(&recv_buf2x,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=myrank_y*PROCESS_Nx+myrank_x+1;

  if(myrank_x==PROCESS_Nx-1){
    dest=myrank_y*PROCESS_Nx+0;
  }

  MPI_Send(&send_buf2x,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  for(k=0;k<2;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][l][m].phi=recv_buf2x[i];
	i+=1;
      }
    }
  }

  i=0;
  for(k=2;k<4;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	send_buf2x[i]=grid[k][l][m].phi;
	i+=1;
      }
    }
  }

  src=myrank_y*PROCESS_Nx+myrank_x+1;

  if(myrank_x==PROCESS_Nx-1){
    src=myrank_y*PROCESS_Nx+0;
  }

  MPI_Irecv(&recv_buf2x,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=myrank_y*PROCESS_Nx+myrank_x-1;

  if(myrank_x==0){
    dest=myrank_y*PROCESS_Nx+PROCESS_Nx-1;
  }

  MPI_Send(&send_buf2x,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  for(k=Grid_Nx+2;k<Grid_Nx+4;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][l][m].phi=recv_buf2x[i];
	i+=1;
      }
    }
  }

  //y direction

  count=2*(Grid_Nx+4)*(Grid_Nz+4);

  i=0;
  for(k=0;k<Grid_Nx+4;k++){
    for(l=Grid_Ny;l<Grid_Ny+2;l++){
      for(m=0;m<Grid_Nz+4;m++){
	send_buf2y[i]=grid[k][l][m].phi;
	i+=1;
      }
    }
  }

  src=(myrank_y-1)*PROCESS_Nx+myrank_x;

  if(myrank_y==0){
    src=(PROCESS_Ny-1)*PROCESS_Nx+myrank_x;
  }

  MPI_Irecv(&recv_buf2y,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=(myrank_y+1)*PROCESS_Nx+myrank_x;

  if(myrank_y==PROCESS_Ny-1){
    dest=0*PROCESS_Nx+myrank_x;
  }

  MPI_Send(&send_buf2y,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  for(k=0;k<Grid_Nx+4;k++){
    for(l=0;l<2;l++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][l][m].phi=recv_buf2y[i];
	i+=1;
      }
    }
  }

  i=0;
  for(k=0;k<Grid_Nx+4;k++){
    for(l=2;l<4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	send_buf2y[i]=grid[k][l][m].phi;
	i+=1;
      }
    }
  }

  src=(myrank_y+1)*PROCESS_Nx+myrank_x;

  if(myrank_y==PROCESS_Ny-1){
    src=0*PROCESS_Nx+myrank_x;
  }

  MPI_Irecv(&recv_buf2y,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=(myrank_y-1)*PROCESS_Nx+myrank_x;

  if(myrank_y==0){
    dest=(PROCESS_Ny-1)*PROCESS_Nx+myrank_x;
  }

  MPI_Send(&send_buf2y,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  for(k=0;k<Grid_Nx+4;k++){
    for(l=Grid_Ny+2;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][l][m].phi=recv_buf2y[i];
	i+=1;
      }
    }
  }

  for(k=2;k<Grid_Nx+2;k++){
    for(l=2;l<Grid_Ny+2;l++){
      for(m=2;m<Grid_Nz+2;m++){
	grid[k][l][m].ex+=(-grid[k+1][l][m].phi+27*grid[k][l][m].phi-27*grid[k-1][l][m].phi+grid[k-2][l][m].phi)/(24.*dx);
	grid[k][l][m].ey+=(-grid[k][l+1][m].phi+27*grid[k][l][m].phi-27*grid[k][l-1][m].phi+grid[k][l-2][m].phi)/(24.*dx);
	grid[k][l][m].ez+=(-grid[k][l][m+1].phi+27*grid[k][l][m].phi-27*grid[k][l][m-1].phi+grid[k][l][m-2].phi)/(24.*dx);
      }
    }
  }

  return(0);
}

int sr_current(const int myrank_x,const int myrank_y)
{  
  int src,dest,tag,count;
  MPI_Status stat;
  MPI_Request request;
  int i,j;
  int k,l,m;

  double send_buf2x[(Grid_Ny+4)*(Grid_Nz+4)*8];
  double send_buf2y[(Grid_Nx+4)*(Grid_Nz+4)*8];
  double recv_buf2x[(Grid_Ny+4)*(Grid_Nz+4)*8];
  double recv_buf2y[(Grid_Nx+4)*(Grid_Nz+4)*8];

  //x direction

  i=0;
  k=Grid_Nx+2;
  for(l=0;l<Grid_Ny+4;l++){
    for(m=0;m<Grid_Nz+4;m++){
      send_buf2x[i]=grid[k][l][m].jix0;
      send_buf2x[i+1]=grid[k][l][m].jiy0;
      send_buf2x[i+2]=grid[k][l][m].jiz0;
      send_buf2x[i+3]=grid[k][l][m].jex0;
      send_buf2x[i+4]=grid[k][l][m].jey0;
      send_buf2x[i+5]=grid[k][l][m].jez0;
      send_buf2x[i+6]=grid[k][l][m].ni;
      send_buf2x[i+7]=grid[k][l][m].ne;
      i+=8;
    }
  }

  tag=1000;

  count=(Grid_Ny+4)*(Grid_Nz+4)*8;

  src=myrank_y*PROCESS_Nx+myrank_x-1;

  if(myrank_x==0){
    src=myrank_y*PROCESS_Nx+PROCESS_Nx-1;
  }

  MPI_Irecv(&recv_buf2x,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=myrank_y*PROCESS_Nx+myrank_x+1;

  if(myrank_x==PROCESS_Nx-1){
    dest=myrank_y*PROCESS_Nx+0;
  }

  MPI_Send(&send_buf2x,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  k=2;
  for(l=0;l<Grid_Ny+4;l++){
    for(m=0;m<Grid_Nz+4;m++){
      grid[k][l][m].jix0+=recv_buf2x[i];
      grid[k][l][m].jiy0+=recv_buf2x[i+1];
      grid[k][l][m].jiz0+=recv_buf2x[i+2];
      grid[k][l][m].jex0+=recv_buf2x[i+3];
      grid[k][l][m].jey0+=recv_buf2x[i+4];
      grid[k][l][m].jez0+=recv_buf2x[i+5];
      grid[k][l][m].ni+=recv_buf2x[i+6];
      grid[k][l][m].ne+=recv_buf2x[i+7];
      i+=8;
    }
  }

  i=0;
  k=1;
  for(l=0;l<Grid_Ny+4;l++){
    for(m=0;m<Grid_Nz+4;m++){
      send_buf2x[i]=grid[k][l][m].jix0;
      send_buf2x[i+1]=grid[k][l][m].jiy0;
      send_buf2x[i+2]=grid[k][l][m].jiz0;
      send_buf2x[i+3]=grid[k][l][m].jex0;
      send_buf2x[i+4]=grid[k][l][m].jey0;
      send_buf2x[i+5]=grid[k][l][m].jez0;
      send_buf2x[i+6]=grid[k][l][m].ni;
      send_buf2x[i+7]=grid[k][l][m].ne;
      i+=8;
    }
  }

  src=myrank_y*PROCESS_Nx+myrank_x+1;

  if(myrank_x==PROCESS_Nx-1){
    src=myrank_y*PROCESS_Nx+0;
  }

  MPI_Irecv(&recv_buf2x,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=myrank_y*PROCESS_Nx+myrank_x-1;

  if(myrank_x==0){
    dest=myrank_y*PROCESS_Nx+PROCESS_Nx-1;
  }

  MPI_Send(&send_buf2x,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  k=Grid_Nx+1;
  for(l=0;l<Grid_Ny+4;l++){
    for(m=0;m<Grid_Nz+4;m++){
      grid[k][l][m].jix0+=recv_buf2x[i];
      grid[k][l][m].jiy0+=recv_buf2x[i+1];
      grid[k][l][m].jiz0+=recv_buf2x[i+2];
      grid[k][l][m].jex0+=recv_buf2x[i+3];
      grid[k][l][m].jey0+=recv_buf2x[i+4];
      grid[k][l][m].jez0+=recv_buf2x[i+5];
      grid[k][l][m].ni+=recv_buf2x[i+6];
      grid[k][l][m].ne+=recv_buf2x[i+7];
      i+=8;
    }
  }

  //y direction

  i=0;
  l=Grid_Ny+2;
  for(k=0;k<Grid_Nx+4;k++){
    for(m=0;m<Grid_Nz+4;m++){
      send_buf2y[i]=grid[k][l][m].jix0;
      send_buf2y[i+1]=grid[k][l][m].jiy0;
      send_buf2y[i+2]=grid[k][l][m].jiz0;
      send_buf2y[i+3]=grid[k][l][m].jex0;
      send_buf2y[i+4]=grid[k][l][m].jey0;
      send_buf2y[i+5]=grid[k][l][m].jez0;
      send_buf2y[i+6]=grid[k][l][m].ni;
      send_buf2y[i+7]=grid[k][l][m].ne;
      i+=8;
    }
  }

  tag=1000;

  count=(Grid_Nx+4)*(Grid_Nz+4)*8;

  src=(myrank_y-1)*PROCESS_Nx+myrank_x;

  if(myrank_y==0){
    src=(PROCESS_Ny-1)*PROCESS_Nx+myrank_x;
  }

  MPI_Irecv(&recv_buf2y,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=(myrank_y+1)*PROCESS_Nx+myrank_x;

  if(myrank_y==PROCESS_Ny-1){
    dest=0*PROCESS_Nx+myrank_x;
  }

  MPI_Send(&send_buf2y,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  l=2;
  for(k=0;k<Grid_Nx+4;k++){
    for(m=0;m<Grid_Nz+4;m++){
      grid[k][l][m].jix0+=recv_buf2y[i];
      grid[k][l][m].jiy0+=recv_buf2y[i+1];
      grid[k][l][m].jiz0+=recv_buf2y[i+2];
      grid[k][l][m].jex0+=recv_buf2y[i+3];
      grid[k][l][m].jey0+=recv_buf2y[i+4];
      grid[k][l][m].jez0+=recv_buf2y[i+5];
      grid[k][l][m].ni+=recv_buf2y[i+6];
      grid[k][l][m].ne+=recv_buf2y[i+7];
      i+=8;
    }
  }

  i=0;
  l=1;
  for(k=0;k<Grid_Nx+4;k++){
    for(m=0;m<Grid_Nz+4;m++){
      send_buf2y[i]=grid[k][l][m].jix0;
      send_buf2y[i+1]=grid[k][l][m].jiy0;
      send_buf2y[i+2]=grid[k][l][m].jiz0;
      send_buf2y[i+3]=grid[k][l][m].jex0;
      send_buf2y[i+4]=grid[k][l][m].jey0;
      send_buf2y[i+5]=grid[k][l][m].jez0;
      send_buf2y[i+6]=grid[k][l][m].ni;
      send_buf2y[i+7]=grid[k][l][m].ne;
      i+=8;
    }
  }

  src=(myrank_y+1)*PROCESS_Nx+myrank_x;

  if(myrank_y==PROCESS_Ny-1){
    src=0*PROCESS_Nx+myrank_x;
  }

  MPI_Irecv(&recv_buf2y,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=(myrank_y-1)*PROCESS_Nx+myrank_x;

  if(myrank_y==0){
    dest=(PROCESS_Ny-1)*PROCESS_Nx+myrank_x;
  }

  MPI_Send(&send_buf2y,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  l=Grid_Ny+1;
  for(k=0;k<Grid_Nx+4;k++){
    for(m=0;m<Grid_Nz+4;m++){
      grid[k][l][m].jix0+=recv_buf2y[i];
      grid[k][l][m].jiy0+=recv_buf2y[i+1];
      grid[k][l][m].jiz0+=recv_buf2y[i+2];
      grid[k][l][m].jex0+=recv_buf2y[i+3];
      grid[k][l][m].jey0+=recv_buf2y[i+4];
      grid[k][l][m].jez0+=recv_buf2y[i+5];
      grid[k][l][m].ni+=recv_buf2y[i+6];
      grid[k][l][m].ne+=recv_buf2y[i+7];
      i+=8;
    }
  }

  return(0);
}

int sr_current2(const int myrank_x,const int myrank_y)
{  
  int src,dest,tag,count;
  MPI_Status stat;
  MPI_Request request;
  int i,j;
  int k,l,m;

  double send_buf2x[2*(Grid_Ny+4)*(Grid_Nz+4)*8];
  double send_buf2y[2*(Grid_Nx+4)*(Grid_Nz+4)*8];
  double recv_buf2x[2*(Grid_Ny+4)*(Grid_Nz+4)*8];
  double recv_buf2y[2*(Grid_Nx+4)*(Grid_Nz+4)*8];

  //x direction

  i=0;
  for(k=Grid_Nx+2;k<Grid_Nx+4;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	send_buf2x[i]=grid[k][l][m].jix0;
	send_buf2x[i+1]=grid[k][l][m].jiy0;
	send_buf2x[i+2]=grid[k][l][m].jiz0;
	send_buf2x[i+3]=grid[k][l][m].jex0;
	send_buf2x[i+4]=grid[k][l][m].jey0;
	send_buf2x[i+5]=grid[k][l][m].jez0;
	send_buf2x[i+6]=grid[k][l][m].ni;
	send_buf2x[i+7]=grid[k][l][m].ne;
	i+=8;
      }
    }
  }

  tag=1000;

  count=2*(Grid_Ny+4)*(Grid_Nz+4)*8;

  src=myrank_y*PROCESS_Nx+myrank_x-1;

  if(myrank_x==0){
    src=myrank_y*PROCESS_Nx+PROCESS_Nx-1;
  }

  MPI_Irecv(&recv_buf2x,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=myrank_y*PROCESS_Nx+myrank_x+1;

  if(myrank_x==PROCESS_Nx-1){
    dest=myrank_y*PROCESS_Nx+0;
  }

  MPI_Send(&send_buf2x,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  for(k=2;k<4;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][l][m].jix0+=recv_buf2x[i];
	grid[k][l][m].jiy0+=recv_buf2x[i+1];
	grid[k][l][m].jiz0+=recv_buf2x[i+2];
	grid[k][l][m].jex0+=recv_buf2x[i+3];
	grid[k][l][m].jey0+=recv_buf2x[i+4];
	grid[k][l][m].jez0+=recv_buf2x[i+5];
	grid[k][l][m].ni+=recv_buf2x[i+6];
	grid[k][l][m].ne+=recv_buf2x[i+7];
	i+=8;
      }
    }
  }

  i=0;
  for(k=0;k<2;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	send_buf2x[i]=grid[k][l][m].jix0;
	send_buf2x[i+1]=grid[k][l][m].jiy0;
	send_buf2x[i+2]=grid[k][l][m].jiz0;
	send_buf2x[i+3]=grid[k][l][m].jex0;
	send_buf2x[i+4]=grid[k][l][m].jey0;
	send_buf2x[i+5]=grid[k][l][m].jez0;
	send_buf2x[i+6]=grid[k][l][m].ni;
	send_buf2x[i+7]=grid[k][l][m].ne;
	i+=8;
      }
    }
  }

  src=myrank_y*PROCESS_Nx+myrank_x+1;

  if(myrank_x==PROCESS_Nx-1){
    src=myrank_y*PROCESS_Nx+0;
  }

  MPI_Irecv(&recv_buf2x,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=myrank_y*PROCESS_Nx+myrank_x-1;

  if(myrank_x==0){
    dest=myrank_y*PROCESS_Nx+PROCESS_Nx-1;
  }

  MPI_Send(&send_buf2x,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  for(k=Grid_Nx;k<Grid_Nx+2;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][l][m].jix0+=recv_buf2x[i];
	grid[k][l][m].jiy0+=recv_buf2x[i+1];
	grid[k][l][m].jiz0+=recv_buf2x[i+2];
	grid[k][l][m].jex0+=recv_buf2x[i+3];
	grid[k][l][m].jey0+=recv_buf2x[i+4];
	grid[k][l][m].jez0+=recv_buf2x[i+5];
	grid[k][l][m].ni+=recv_buf2x[i+6];
	grid[k][l][m].ne+=recv_buf2x[i+7];
	i+=8;
      }
    }
  }
  
  //y direction
  
  i=0;
  for(k=0;k<Grid_Nx+4;k++){
    for(l=Grid_Ny+2;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	send_buf2y[i]=grid[k][l][m].jix0;
	send_buf2y[i+1]=grid[k][l][m].jiy0;
	send_buf2y[i+2]=grid[k][l][m].jiz0;
	send_buf2y[i+3]=grid[k][l][m].jex0;
	send_buf2y[i+4]=grid[k][l][m].jey0;
	send_buf2y[i+5]=grid[k][l][m].jez0;
	send_buf2y[i+6]=grid[k][l][m].ni;
	send_buf2y[i+7]=grid[k][l][m].ne;
	i+=8;
      }
    }
  }

  tag=1000;

  count=2*(Grid_Nx+4)*(Grid_Nz+4)*8;

  src=(myrank_y-1)*PROCESS_Nx+myrank_x;

  if(myrank_y==0){
    src=(PROCESS_Ny-1)*PROCESS_Nx+myrank_x;
  }

  MPI_Irecv(&recv_buf2y,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=(myrank_y+1)*PROCESS_Nx+myrank_x;

  if(myrank_y==PROCESS_Ny-1){
    dest=0*PROCESS_Nx+myrank_x;
  }

  MPI_Send(&send_buf2y,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  for(k=0;k<Grid_Nx+4;k++){
    for(l=2;l<4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][l][m].jix0+=recv_buf2y[i];
	grid[k][l][m].jiy0+=recv_buf2y[i+1];
	grid[k][l][m].jiz0+=recv_buf2y[i+2];
	grid[k][l][m].jex0+=recv_buf2y[i+3];
	grid[k][l][m].jey0+=recv_buf2y[i+4];
	grid[k][l][m].jez0+=recv_buf2y[i+5];
	grid[k][l][m].ni+=recv_buf2y[i+6];
	grid[k][l][m].ne+=recv_buf2y[i+7];
	i+=8;
      }
    }
  }

  i=0;
  for(k=0;k<Grid_Nx+4;k++){
    for(l=0;l<2;l++){
      for(m=0;m<Grid_Nz+4;m++){
	send_buf2y[i]=grid[k][l][m].jix0;
	send_buf2y[i+1]=grid[k][l][m].jiy0;
	send_buf2y[i+2]=grid[k][l][m].jiz0;
	send_buf2y[i+3]=grid[k][l][m].jex0;
	send_buf2y[i+4]=grid[k][l][m].jey0;
	send_buf2y[i+5]=grid[k][l][m].jez0;
	send_buf2y[i+6]=grid[k][l][m].ni;
	send_buf2y[i+7]=grid[k][l][m].ne;
	i+=8;
      }
    }
  }

  src=(myrank_y+1)*PROCESS_Nx+myrank_x;

  if(myrank_y==PROCESS_Ny-1){
    src=0*PROCESS_Nx+myrank_x;
  }

  MPI_Irecv(&recv_buf2y,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=(myrank_y-1)*PROCESS_Nx+myrank_x;

  if(myrank_y==0){
    dest=(PROCESS_Ny-1)*PROCESS_Nx+myrank_x;
  }

  MPI_Send(&send_buf2y,count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  i=0;
  for(k=0;k<Grid_Nx+4;k++){
    for(l=Grid_Ny;l<Grid_Ny+2;l++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][l][m].jix0+=recv_buf2y[i];
	grid[k][l][m].jiy0+=recv_buf2y[i+1];
	grid[k][l][m].jiz0+=recv_buf2y[i+2];
	grid[k][l][m].jex0+=recv_buf2y[i+3];
	grid[k][l][m].jey0+=recv_buf2y[i+4];
	grid[k][l][m].jez0+=recv_buf2y[i+5];
	grid[k][l][m].ni+=recv_buf2y[i+6];
	grid[k][l][m].ne+=recv_buf2y[i+7];
	i+=8;
      }
    }
  }

  return(0);
}

int sr_particle(Particle *ion[],Particle *ion_start[],Particle *electron[],Particle *electron_start[],const int myrank_x,const int myrank_y)
{
  int i;
  int count_i_fwd[THREAD_N];
  int count_i_bak[THREAD_N];
  int count_e_fwd[THREAD_N];
  int count_e_bak[THREAD_N];
  Particle pack_i_fwd[THREAD_N][PACK_N];
  Particle pack_i_bak[THREAD_N][PACK_N];
  Particle pack_e_fwd[THREAD_N][PACK_N];
  Particle pack_e_bak[THREAD_N][PACK_N];

  const double dx2=2.*dx;
  const double gx2=(Grid_Nx+2)*dx;
  const double gy2=(Grid_Ny+2)*dx;
  const double gz2=(Grid_Nz+2)*dx;

  //x direction

#pragma omp parallel for num_threads(active_thread)
  for(i=0;i<THREAD_N;i++){
    count_i_fwd[i]=0;
    count_i_bak[i]=0;
    count_e_fwd[i]=0;
    count_e_bak[i]=0;

    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){
      if(ion[i]->z<gz2&&ion[i]->z>=dx2){
	if(ion[i]->x>=gx2){
	  //pack_i_fwd[i][count_i_fwd[i]]=*ion[i];
	  memcpy(&pack_i_fwd[i][count_i_fwd[i]],ion[i],sizeof(Particle));
	  count_i_fwd[i]++;
	  del_particle_i(ion,i,ion_start);
	}else if(ion[i]->x<dx2){
	  //pack_i_bak[i][count_i_bak[i]]=*ion[i];
	  memcpy(&pack_i_bak[i][count_i_bak[i]],ion[i],sizeof(Particle));
	  count_i_bak[i]++;
	  del_particle_i(ion,i,ion_start);
	}
      }else{
	del_particle_i(ion,i,ion_start);
      }
      ion[i]=ion[i]->next_particle;
    }
    if(ion[i]->z<gz2&&ion[i]->z>=dx2){
      if(ion[i]->x>=gx2){
	//pack_i_fwd[i][count_i_fwd[i]]=*ion[i];
	memcpy(&pack_i_fwd[i][count_i_fwd[i]],ion[i],sizeof(Particle));
	count_i_fwd[i]++;
	del_particle_i(ion,i,ion_start);
      }else if(ion[i]->x<dx2){
	//pack_i_bak[i][count_i_bak[i]]=*ion[i];
	memcpy(&pack_i_bak[i][count_i_bak[i]],ion[i],sizeof(Particle));
	count_i_bak[i]++;
	del_particle_i(ion,i,ion_start);
      }
    }else{
      del_particle_i(ion,i,ion_start);
    }
    
    electron[i]=electron_start[i];
    while(electron[i]->next_particle!=NULL){
      if(electron[i]->z<gz2&&electron[i]->z>=dx2){
	if(electron[i]->x>=gx2){
	  //pack_e_fwd[i][count_e_fwd[i]]=*electron[i];
	  memcpy(&pack_e_fwd[i][count_e_fwd[i]],electron[i],sizeof(Particle));
	  count_e_fwd[i]++;
	  del_particle_e(electron,i,electron_start);
	}else if(electron[i]->x<dx2){
	  //pack_e_bak[i][count_e_bak[i]]=*electron[i];
	  memcpy(&pack_e_bak[i][count_e_bak[i]],electron[i],sizeof(Particle));
	  count_e_bak[i]++;
	  del_particle_e(electron,i,electron_start);
	}
      }else{
	del_particle_e(electron,i,electron_start);
      }
      electron[i]=electron[i]->next_particle;
    }
    if(electron[i]->z<gz2&&electron[i]->z>=dx2){
      if(electron[i]->x>=gx2){
	//pack_e_fwd[i][count_e_fwd[i]]=*electron[i];
	memcpy(&pack_e_fwd[i][count_e_fwd[i]],electron[i],sizeof(Particle));
	count_e_fwd[i]++;
	del_particle_e(electron,i,electron_start);
      }else if(electron[i]->x<dx2){
	//pack_e_bak[i][count_e_bak[i]]=*electron[i];
	memcpy(&pack_e_bak[i][count_e_bak[i]],electron[i],sizeof(Particle));
	count_e_bak[i]++;
	del_particle_e(electron,i,electron_start);
      }
    }else{
      del_particle_e(electron,i,electron_start);
    }
  }
  
  if(myrank_x==0){
    for(i=0;i<THREAD_N;i++){
      count_i_bak[i]=0;
      count_e_bak[i]=0;
    }
  }else if(myrank_x==PROCESS_Nx-1){
    for(i=0;i<THREAD_N;i++){
      count_i_fwd[i]=0;
      count_e_fwd[i]=0;
    }
  }

  sr_particle_x(pack_i_fwd,count_i_fwd,ion,ion_start,myrank_x,myrank_y,1);
  sr_particle_x(pack_i_bak,count_i_bak,ion,ion_start,myrank_x,myrank_y,-1);
  sr_particle_x(pack_e_fwd,count_e_fwd,electron,electron_start,myrank_x,myrank_y,1);
  sr_particle_x(pack_e_bak,count_e_bak,electron,electron_start,myrank_x,myrank_y,-1);

  //y direction

#pragma omp parallel for num_threads(active_thread)
  for(i=0;i<THREAD_N;i++){
    count_i_fwd[i]=0;
    count_i_bak[i]=0;
    count_e_fwd[i]=0;
    count_e_bak[i]=0;

    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){
      if(ion[i]->z<gz2&&ion[i]->z>=dx2){
	if(ion[i]->y>=gy2){
	  //pack_i_fwd[i][count_i_fwd[i]]=*ion[i];
	  memcpy(&pack_i_fwd[i][count_i_fwd[i]],ion[i],sizeof(Particle));
	  count_i_fwd[i]++;
	  del_particle_i(ion,i,ion_start);
	}else if(ion[i]->y<dx2){
	  //pack_i_bak[i][count_i_bak[i]]=*ion[i];
	  memcpy(&pack_i_bak[i][count_i_bak[i]],ion[i],sizeof(Particle));
	  count_i_bak[i]++;
	  del_particle_i(ion,i,ion_start);
	}
      }else{
	del_particle_i(ion,i,ion_start);
      }
      ion[i]=ion[i]->next_particle;
    }
    if(ion[i]->z<gz2&&ion[i]->z>=dx2){
      if(ion[i]->y>=gy2){
	//pack_i_fwd[i][count_i_fwd[i]]=*ion[i];
	memcpy(&pack_i_fwd[i][count_i_fwd[i]],ion[i],sizeof(Particle));
	count_i_fwd[i]++;
	del_particle_i(ion,i,ion_start);
      }else if(ion[i]->y<dx2){
	//pack_i_bak[i][count_i_bak[i]]=*ion[i];
	memcpy(&pack_i_bak[i][count_i_bak[i]],ion[i],sizeof(Particle));
	count_i_bak[i]++;
	del_particle_i(ion,i,ion_start);
      }
    }else{
      del_particle_i(ion,i,ion_start);
    }
    
    electron[i]=electron_start[i];
    while(electron[i]->next_particle!=NULL){
      if(electron[i]->z<gz2&&electron[i]->z>=dx2){
	if(electron[i]->y>=gy2){
	  //pack_e_fwd[i][count_e_fwd[i]]=*electron[i];
	  memcpy(&pack_e_fwd[i][count_e_fwd[i]],electron[i],sizeof(Particle));
	  count_e_fwd[i]++;
	  del_particle_e(electron,i,electron_start);
	}else if(electron[i]->y<dx2){
	  //pack_e_bak[i][count_e_bak[i]]=*electron[i];
	  memcpy(&pack_e_bak[i][count_e_bak[i]],electron[i],sizeof(Particle));
	  count_e_bak[i]++;
	  del_particle_e(electron,i,electron_start);
	}
      }else{
	del_particle_e(electron,i,electron_start);
      }
      electron[i]=electron[i]->next_particle;
    }
    if(electron[i]->z<gz2&&electron[i]->z>=dx2){
      if(electron[i]->y>=gy2){
	//pack_e_fwd[i][count_e_fwd[i]]=*electron[i];
	memcpy(&pack_e_fwd[i][count_e_fwd[i]],electron[i],sizeof(Particle));
	count_e_fwd[i]++;
	del_particle_e(electron,i,electron_start);
      }else if(electron[i]->y<dx2){
	//pack_e_bak[i][count_e_bak[i]]=*electron[i];
	memcpy(&pack_e_bak[i][count_e_bak[i]],electron[i],sizeof(Particle));
	count_e_bak[i]++;
	del_particle_e(electron,i,electron_start);
      }
    }else{
      del_particle_e(electron,i,electron_start);
    }
  }
  
  if(myrank_y==0){
    for(i=0;i<THREAD_N;i++){
      count_i_bak[i]=0;
      count_e_bak[i]=0;
    }
  }else if(myrank_y==PROCESS_Ny-1){
    for(i=0;i<THREAD_N;i++){
      count_i_fwd[i]=0;
      count_e_fwd[i]=0;
    }
  }

  sr_particle_y(pack_i_fwd,count_i_fwd,ion,ion_start,myrank_x,myrank_y,1);
  sr_particle_y(pack_i_bak,count_i_bak,ion,ion_start,myrank_x,myrank_y,-1);
  sr_particle_y(pack_e_fwd,count_e_fwd,electron,electron_start,myrank_x,myrank_y,1);
  sr_particle_y(pack_e_bak,count_e_bak,electron,electron_start,myrank_x,myrank_y,-1);

  return(0);
}

int sr_particle_x(const Particle pack_particle[][PACK_N],int count_p[],Particle *particle[],Particle *particle_start[],const int myrank_x,const int myrank_y,const int flag)
{  
  int src,dest,tag,count;
  MPI_Status stat[THREAD_N];
  MPI_Request request[THREAD_N];
  int i,j;
  int count_p_r[THREAD_N];
  static Particle pack_particle_r[THREAD_N][PACK_N];

  tag=1000;

  for(i=0;i<THREAD_N;i++){
    count_p_r[i]=0;
  }

  if(flag==1){
    for(i=0;i<THREAD_N;i++){
      src=myrank_y*PROCESS_Nx+myrank_x-1;
      
      if(myrank_x==0){
	src=myrank_y*PROCESS_Nx+PROCESS_Nx-1;
      }
      
      MPI_Irecv(&count_p_r[i],1,MPI_INT,src,tag,MPI_COMM_WORLD,&request[i]);
      
      dest=myrank_y*PROCESS_Nx+myrank_x+1;
      
      if(myrank_x==PROCESS_Nx-1){
	dest=myrank_y*PROCESS_Nx+0;
      }
      
      MPI_Send(&count_p[i],1,MPI_INT,dest,tag,MPI_COMM_WORLD);
      
      MPI_Wait(&request[i],&stat[i]);
    }

    for(i=0;i<THREAD_N;i++){
      count=count_p_r[i]*sizeof(Particle)/sizeof(int);
      
      src=myrank_y*PROCESS_Nx+myrank_x-1;
      
      if(myrank_x==0){
	src=myrank_y*PROCESS_Nx+PROCESS_Nx-1;
      }
      
      MPI_Irecv(&pack_particle_r[i],count,MPI_INT,src,tag,MPI_COMM_WORLD,&request[i]);

      count=count_p[i]*sizeof(Particle)/sizeof(int);

      dest=myrank_y*PROCESS_Nx+myrank_x+1;
      
      if(myrank_x==PROCESS_Nx-1){
	dest=myrank_y*PROCESS_Nx+0;
      }
      
      MPI_Send(&pack_particle[i],count,MPI_INT,dest,tag,MPI_COMM_WORLD);
      
      MPI_Wait(&request[i],&stat[i]);
    }
  }else if(flag==-1){
    for(i=0;i<THREAD_N;i++){
      src=myrank_y*PROCESS_Nx+myrank_x+1;
      
      if(myrank_x==PROCESS_Nx-1){
	src=myrank_y*PROCESS_Nx+0;
      }
      
      MPI_Irecv(&count_p_r[i],1,MPI_INT,src,tag,MPI_COMM_WORLD,&request[i]);

      dest=myrank_y*PROCESS_Nx+myrank_x-1;
      
      if(myrank_x==0){
	dest=myrank_y*PROCESS_Nx+PROCESS_Nx-1;
      }
      
      MPI_Send(&count_p[i],1,MPI_INT,dest,tag,MPI_COMM_WORLD);
      
      MPI_Wait(&request[i],&stat[i]);
    }

    for(i=0;i<THREAD_N;i++){
      count=count_p_r[i]*sizeof(Particle)/sizeof(int);

      src=myrank_y*PROCESS_Nx+myrank_x+1;
      
      if(myrank_x==PROCESS_Nx-1){
	src=myrank_y*PROCESS_Nx+0;
      }
      
      MPI_Irecv(&pack_particle_r[i],count,MPI_INT,src,tag,MPI_COMM_WORLD,&request[i]);

      count=count_p[i]*sizeof(Particle)/sizeof(int);
 
      dest=myrank_y*PROCESS_Nx+myrank_x-1;
      
      if(myrank_x==0){
	dest=myrank_y*PROCESS_Nx+PROCESS_Nx-1;
      }

      MPI_Send(&pack_particle[i],count,MPI_INT,dest,tag,MPI_COMM_WORLD);

      MPI_Wait(&request[i],&stat[i]);
    }
  }

#pragma omp parallel for private(j) num_threads(active_thread)
  for(i=0;i<THREAD_N;i++){
    particle[i]=particle_start[i];
    while(particle[i]->next_particle!=NULL){
      particle[i]=particle[i]->next_particle;
    }

    for(j=0;j<count_p_r[i];j++){
      if((particle[i]->next_particle=(Particle*)malloc(sizeof(Particle)))!=NULL){
	particle[i]->next_particle->prev_particle=particle[i];
	particle[i]=particle[i]->next_particle;

	if(flag==1){
	  particle[i]->x=pack_particle_r[i][j].x-Grid_Nx*dx;
	}else if(flag==-1){
	  particle[i]->x=pack_particle_r[i][j].x+Grid_Nx*dx;
	}
   
	particle[i]->y=pack_particle_r[i][j].y;
	particle[i]->z=pack_particle_r[i][j].z;
	particle[i]->vx=pack_particle_r[i][j].vx;
	particle[i]->vy=pack_particle_r[i][j].vy;
	particle[i]->vz=pack_particle_r[i][j].vz;
	particle[i]->n=pack_particle_r[i][j].n;
	particle[i]->flag=pack_particle_r[i][j].flag;

	particle[i]->next_particle=NULL;
      }
    }
  }
  
  return(0);
}

int sr_particle_y(const Particle pack_particle[][PACK_N],int count_p[],Particle *particle[],Particle *particle_start[],const int myrank_x,const int myrank_y,const int flag)
{  
  int src,dest,tag,count;
  MPI_Status stat[THREAD_N];
  MPI_Request request[THREAD_N];
  int i,j;
  int count_p_r[THREAD_N];
  static Particle pack_particle_r[THREAD_N][PACK_N];

  tag=1000;

  for(i=0;i<THREAD_N;i++){
    count_p_r[i]=0;
  }

  if(flag==1){
    for(i=0;i<THREAD_N;i++){
      src=(myrank_y-1)*PROCESS_Nx+myrank_x;
      
      if(myrank_y==0){
	src=(PROCESS_Ny-1)*PROCESS_Nx+myrank_x;
      }
      
      MPI_Irecv(&count_p_r[i],1,MPI_INT,src,tag,MPI_COMM_WORLD,&request[i]);
      
      dest=(myrank_y+1)*PROCESS_Nx+myrank_x;
      
      if(myrank_y==PROCESS_Ny-1){
	dest=0*PROCESS_Nx+myrank_x;
      }
      
      MPI_Send(&count_p[i],1,MPI_INT,dest,tag,MPI_COMM_WORLD);
      
      MPI_Wait(&request[i],&stat[i]);
    }

    for(i=0;i<THREAD_N;i++){
      count=count_p_r[i]*sizeof(Particle)/sizeof(int);
      
      src=(myrank_y-1)*PROCESS_Nx+myrank_x;
      
      if(myrank_y==0){
	src=(PROCESS_Ny-1)*PROCESS_Nx+myrank_x;
      }
      
      MPI_Irecv(&pack_particle_r[i],count,MPI_INT,src,tag,MPI_COMM_WORLD,&request[i]);

      count=count_p[i]*sizeof(Particle)/sizeof(int);

      dest=(myrank_y+1)*PROCESS_Nx+myrank_x;
      
      if(myrank_y==PROCESS_Ny-1){
	dest=0*PROCESS_Nx+myrank_x;
      }
      
      MPI_Send(&pack_particle[i],count,MPI_INT,dest,tag,MPI_COMM_WORLD);
      
      MPI_Wait(&request[i],&stat[i]);
    }
  }else if(flag==-1){
    for(i=0;i<THREAD_N;i++){
      src=(myrank_y+1)*PROCESS_Nx+myrank_x;
      
      if(myrank_y==PROCESS_Ny-1){
	src=0*PROCESS_Nx+myrank_x;
      }
      
      MPI_Irecv(&count_p_r[i],1,MPI_INT,src,tag,MPI_COMM_WORLD,&request[i]);

      dest=(myrank_y-1)*PROCESS_Nx+myrank_x;
      
      if(myrank_y==0){
	dest=(PROCESS_Ny-1)*PROCESS_Nx+myrank_x;
      }
      
      MPI_Send(&count_p[i],1,MPI_INT,dest,tag,MPI_COMM_WORLD);
      
      MPI_Wait(&request[i],&stat[i]);
    }

    for(i=0;i<THREAD_N;i++){
      count=count_p_r[i]*sizeof(Particle)/sizeof(int);

      src=(myrank_y+1)*PROCESS_Nx+myrank_x;
      
      if(myrank_y==PROCESS_Ny-1){
	src=0*PROCESS_Nx+myrank_x;
      }

      MPI_Irecv(&pack_particle_r[i],count,MPI_INT,src,tag,MPI_COMM_WORLD,&request[i]);

      count=count_p[i]*sizeof(Particle)/sizeof(int);
 
      dest=(myrank_y-1)*PROCESS_Nx+myrank_x;
      
      if(myrank_y==0){
	dest=(PROCESS_Ny-1)*PROCESS_Nx+myrank_x;
      }

      MPI_Send(&pack_particle[i],count,MPI_INT,dest,tag,MPI_COMM_WORLD);

      MPI_Wait(&request[i],&stat[i]);
    }
  }

#pragma omp parallel for private(j) num_threads(active_thread)
  for(i=0;i<THREAD_N;i++){
    particle[i]=particle_start[i];
    while(particle[i]->next_particle!=NULL){
      particle[i]=particle[i]->next_particle;
    }

    for(j=0;j<count_p_r[i];j++){
      if((particle[i]->next_particle=(Particle*)malloc(sizeof(Particle)))!=NULL){
	particle[i]->next_particle->prev_particle=particle[i];
	particle[i]=particle[i]->next_particle;

	if(flag==1){
	  particle[i]->y=pack_particle_r[i][j].y-Grid_Ny*dx;
	}else if(flag==-1){
	  particle[i]->y=pack_particle_r[i][j].y+Grid_Ny*dx;
	}
   
	particle[i]->x=pack_particle_r[i][j].x;
	particle[i]->z=pack_particle_r[i][j].z;
	particle[i]->vx=pack_particle_r[i][j].vx;
	particle[i]->vy=pack_particle_r[i][j].vy;
	particle[i]->vz=pack_particle_r[i][j].vz;
	particle[i]->n=pack_particle_r[i][j].n;
	particle[i]->flag=pack_particle_r[i][j].flag;

	particle[i]->next_particle=NULL;
      }
    }
  }
  
  return(0);
}

int renew_grid()
{
  int i,k,l,m;

#pragma omp parallel for private(l,m)
  for(k=0;k<Grid_Nx+4;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){    
	grid[k][l][m].jix0=0.;
	grid[k][l][m].jiy0=0.;
	grid[k][l][m].jiz0=0.;
	
	grid[k][l][m].jex0=0.;
	grid[k][l][m].jey0=0.;
	grid[k][l][m].jez0=0.;
	
	grid[k][l][m].ni=0.;
	grid[k][l][m].ne=0.;

	grid[k][l][m].phi=0.;
      }
    }
  }

#pragma omp parallel for private(k,l,m)
  for(i=0;i<THREAD_N;i++){
    for(k=0;k<Grid_Nx+4;k++){
      for(l=0;l<Grid_Ny+4;l++){
	for(m=0;m<Grid_Nz+4;m++){    
	  grid_thread[i][k][l][m].jix0=0.;
	  grid_thread[i][k][l][m].jiy0=0.;
	  grid_thread[i][k][l][m].jiz0=0.;
	  
	  grid_thread[i][k][l][m].jex0=0.;
	  grid_thread[i][k][l][m].jey0=0.;
	  grid_thread[i][k][l][m].jez0=0.;
	  
	  grid_thread[i][k][l][m].ni=0.;
	  grid_thread[i][k][l][m].ne=0.;
	}
      }
    }
  }

  return(0);
}

int renew_b4_2()
{
  int k,l,m;
  const double temp=1./e0*dt*0.5*0.5;
  const double dth=0.5*dt;
  const double temp_24=1./24.;
  const double inv_dx=1./dx;
  
#pragma omp parallel for private(l,m)
  for(k=2;k<Grid_Nx+2;k++){
    for(l=2;l<Grid_Ny+2;l++){
      for(m=2;m<Grid_Nz+2;m++){
	grid[k][l][m].bx=grid[k][l][m].b0x
	  +(1.-sigma[2*k+1][2*l][2*m]*temp)/(1.+sigma[2*k+1][2*l][2*m]*temp)*(grid[k][l][m].bx-grid[k][l][m].b0x)
	  -dth/(1.+sigma[2*k+1][2*l][2*m]*temp)
	  *temp_24*(27.*(grid[k][l][m].ez-grid[k][l-1][m].ez-grid[k][l][m].ey+grid[k][l][m-1].ey)-(grid[k][l+1][m].ez-grid[k][l-2][m].ez-grid[k][l][m+1].ey+grid[k][l][m-2].ey))*inv_dx;
	
	grid[k][l][m].by=grid[k][l][m].b0y
	  +(1.-sigma[2*k][2*l+1][2*m]*temp)/(1.+sigma[2*k][2*l+1][2*m]*temp)*(grid[k][l][m].by-grid[k][l][m].b0y)
	  -dth/(1.+sigma[2*k][2*l+1][2*m]*temp)
	  *temp_24*(27.*(grid[k][l][m].ex-grid[k][l][m-1].ex-grid[k][l][m].ez+grid[k-1][l][m].ez)-(grid[k][l][m+1].ex-grid[k][l][m-2].ex-grid[k+1][l][m].ez+grid[k-2][l][m].ez))*inv_dx;
	
	grid[k][l][m].bz=grid[k][l][m].b0z
	  +(1.-sigma[2*k][2*l][2*m+1]*temp)/(1.+sigma[2*k][2*l][2*m+1]*temp)*(grid[k][l][m].bz-grid[k][l][m].b0z)
	  -dth/(1.+sigma[2*k][2*l][2*m+1]*temp)
	  *temp_24*(27.*(grid[k][l][m].ey-grid[k-1][l][m].ey-grid[k][l][m].ex+grid[k][l-1][m].ex)-(grid[k+1][l][m].ey-grid[k-2][l][m].ey-grid[k][l+1][m].ex+grid[k][l-2][m].ex))*inv_dx;
      }
    }
  }

  return(0);
}

int renew_e4_2()
{
  int k,l,m;
  const double temp=C*C*dt;
  const double temp_24=1./24.;
  const double inv_dx=1./dx;

#pragma omp parallel for private(l,m)
  for(k=2;k<Grid_Nx+2;k++){
    for(l=2;l<Grid_Ny+2;l++){
      for(m=2;m<Grid_Nz+2;m++){
	grid[k][l][m].ex=(1.-0.5*mu0*sigma[2*k][2*l+1][2*m+1]*temp)/(1.+0.5*mu0*sigma[2*k][2*l+1][2*m+1]*temp)*grid[k][l][m].ex
	  +temp/(1.+0.5*mu0*sigma[2*k][2*l+1][2*m+1]*temp)
	  *(temp_24*(27.*((grid[k][l+1][m].bz-grid[k][l+1][m].b0z-grid[k][l][m].bz+grid[k][l][m].b0z)
			  -(grid[k][l][m+1].by-grid[k][l][m+1].b0y-grid[k][l][m].by+grid[k][l][m].b0y))
		     -((grid[k][l+2][m].bz-grid[k][l+2][m].b0z-grid[k][l-1][m].bz+grid[k][l-1][m].b0z)
		       -(grid[k][l][m+2].by-grid[k][l][m+2].b0y-grid[k][l][m-1].by+grid[k][l][m-1].b0y)))*inv_dx
	    -mu0*(grid[k][l][m].jix0-grid[k][l][m].jex0));
	
	grid[k][l][m].ey=(1.-0.5*mu0*sigma[2*k+1][2*l][2*m+1]*temp)/(1.+0.5*mu0*sigma[2*k+1][2*l][2*m+1]*temp)*grid[k][l][m].ey
	  +temp/(1.+0.5*mu0*sigma[2*k+1][2*l][2*m+1]*temp)
	  *(temp_24*(27.*((grid[k][l][m+1].bx-grid[k][l][m+1].b0x-grid[k][l][m].bx+grid[k][l][m].b0x)
			  -(grid[k+1][l][m].bz-grid[k+1][l][m].b0z-grid[k][l][m].bz+grid[k][l][m].b0z))
		     -((grid[k][l][m+2].bx-grid[k][l][m+2].b0x-grid[k][l][m-1].bx+grid[k][l][m-1].b0x)
		       -(grid[k+2][l][m].bz-grid[k+2][l][m].b0z-grid[k-1][l][m].bz+grid[k-1][l][m].b0z)))*inv_dx
	    -mu0*(grid[k][l][m].jiy0-grid[k][l][m].jey0));
	
	grid[k][l][m].ez=(1.-0.5*mu0*sigma[2*k+1][2*l+1][2*m]*temp)/(1.+0.5*mu0*sigma[2*k+1][2*l+1][2*m]*temp)*grid[k][l][m].ez
	  +temp/(1.+0.5*mu0*sigma[2*k+1][2*l+1][2*m]*temp)
	  *(temp_24*(27.*((grid[k+1][l][m].by-grid[k+1][l][m].b0y-grid[k][l][m].by+grid[k][l][m].b0y)
			  -(grid[k][l+1][m].bx-grid[k][l+1][m].b0x-grid[k][l][m].bx+grid[k][l][m].b0x))
		     -((grid[k+2][l][m].by-grid[k+2][l][m].b0y-grid[k-1][l][m].by+grid[k-1][l][m].b0y)
		       -(grid[k][l+2][m].bx-grid[k][l+2][m].b0x-grid[k][l-1][m].bx+grid[k][l-1][m].b0x)))*inv_dx
	    -mu0*(grid[k][l][m].jiz0-grid[k][l][m].jez0));
      }
    }
  }

  return(0);
}

inline int cal_track(Particle *p,const double m)
{
  double ux,uy,uz;
  double uox,uoy,uoz;
  double upx,upy,upz;
  double ex,ey,ez,bx,by,bz;
  const double T=0.5*q*dt/m;
  double S;

  shape_func_v2(p,&ex,&ey,&ez,&bx,&by,&bz);
  
  ex+=V*IMF_y;
  ey+=-V*IMF_x;
  
  S=2.*T/(1.+T*T*(bx*bx+by*by+bz*bz));
  
  ux=p->vx+ex*T;
  uy=p->vy+ey*T;
  uz=p->vz+ez*T;
  
  uox=ux+T*(uy*bz-uz*by);
  uoy=uy+T*(uz*bx-ux*bz);
  uoz=uz+T*(ux*by-uy*bx);
  
  p->vx=ux+S*(uoy*bz-uoz*by)+ex*T;
  p->vy=uy+S*(uoz*bx-uox*bz)+ey*T;
  p->vz=uz+S*(uox*by-uoy*bx)+ez*T;
  
  p->x+=p->vx*dt*0.5;
  p->y+=p->vy*dt*0.5;
  p->z+=p->vz*dt*0.5;
  
  p->n=p->n;
  p->flag=p->flag;
  p->next_particle=p->next_particle;
  p->prev_particle=p->prev_particle;

  return(0);
}

inline int cal_track2(Particle *p,const double m,const int myrank_x,const int myrank_y,const int c)
{
  double ux,uy,uz;
  double uox,uoy,uoz;
  double upx,upy,upz;
  double ex,ey,ez,bx,by,bz;
  const double T=0.5*q*dt/m;
  double S;
  double x,y,z,r,w;

  w=Pi/50000.;

  shape_func_v2(p,&ex,&ey,&ez,&bx,&by,&bz);
  
  ex+=V*IMF_y;
  ey+=-V*IMF_x;

  if(c<50000){
    x=p->x+(myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
    y=p->y+(myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
    z=p->z+(-(Grid_Nz+4)/2)*dx;
	  
    r=sqrt(x*x+y*y+z*z);
	  
    bx+=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5))+IMF_x;
    by+=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5))+IMF_y;
    bz+=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5))+IMF_z;
  }else{
    x=p->x+(myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
    y=p->y+(myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
    z=p->z+(-(Grid_Nz+4)/2)*dx;
	  
    r=sqrt(x*x+y*y+z*z);
	  
    bx+=mu0*I*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5))+IMF_x;
    by+=mu0*I*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5))+IMF_y;
    bz+=mu0*I*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5))+IMF_z;
  }

  S=2.*T/(1.+T*T*(bx*bx+by*by+bz*bz));
  
  ux=p->vx+ex*T;
  uy=p->vy+ey*T;
  uz=p->vz+ez*T;
  
  uox=ux+T*(uy*bz-uz*by);
  uoy=uy+T*(uz*bx-ux*bz);
  uoz=uz+T*(ux*by-uy*bx);
  
  p->vx=ux+S*(uoy*bz-uoz*by)+ex*T;
  p->vy=uy+S*(uoz*bx-uox*bz)+ey*T;
  p->vz=uz+S*(uox*by-uoy*bx)+ez*T;
  
  p->x+=p->vx*dt*0.5;
  p->y+=p->vy*dt*0.5;
  p->z+=p->vz*dt*0.5;
  
  p->n=p->n;
  p->flag=p->flag;
  p->next_particle=p->next_particle;
  p->prev_particle=p->prev_particle;

  return(0);
}

inline int cal_track3(Particle *p,const double m,const int myrank_x,const int myrank_y)
{
  double ux,uy,uz;
  double uox,uoy,uoz;
  double upx,upy,upz;
  double ex,ey,ez,bx,by,bz;
  const double T=0.5*q*dt/m;
  double S;
  double Vox,Voy,Voz;

  shape_func_v2(p,&ex,&ey,&ez,&bx,&by,&bz);
  
  ex+=V*IMF_y;
  ey+=-V*IMF_x;
  
  S=2.*T/(1.+T*T*(bx*bx+by*by+bz*bz));

  Vox=-(p->z+(-Grid_Nz/2-2)*dx)*omega_y-(p->y+(myrank_y*Grid_Ny-PROCESS_Ny*Grid_Ny/2-2)*dx)*omega_z;
  Voy=(p->x+(myrank_x*Grid_Nx-PROCESS_Nx*Grid_Nx/2-2)*dx)*omega_z;
  Voz=(p->x+(myrank_x*Grid_Nx-PROCESS_Nx*Grid_Nx/2-2)*dx)*omega_y;

  ux=p->vx+ex*T+Vox;
  uy=p->vy+ey*T+Voy;
  uz=p->vz+ez*T+Voz;
  
  uox=ux+T*(uy*bz-uz*by);
  uoy=uy+T*(uz*bx-ux*bz);
  uoz=uz+T*(ux*by-uy*bx);
  
  p->vx=ux+S*(uoy*bz-uoz*by)+ex*T-Vox;
  p->vy=uy+S*(uoz*bx-uox*bz)+ey*T-Voy;
  p->vz=uz+S*(uox*by-uoy*bx)+ez*T-Voz;
  
  p->x+=p->vx*dt*0.5;
  p->y+=p->vy*dt*0.5;
  p->z+=p->vz*dt*0.5;
  
  p->n=p->n;
  p->flag=p->flag;
  p->next_particle=p->next_particle;
  p->prev_particle=p->prev_particle;

  return(0);
}

inline int cal_track4(Particle *p,const double m,const int myrank_x,const int myrank_y)
{
  double ux,uy,uz;
  double uox,uoy,uoz;
  double upx,upy,upz;
  double ex,ey,ez,bx,by,bz;
  double bux,buy,buz;
  double gamma1,gamma2;
  const double T=0.5*q*dt/m;
  double S;

  shape_func_v2(p,&ex,&ey,&ez,&bx,&by,&bz);
  
  ex+=V*IMF_y;
  ey+=-V*IMF_x;

  gamma1=C/sqrt(C*C-p->vx*p->vx-p->vy*p->vy-p->vz*p->vz);

  if(isnan(gamma1)){
    gamma1=0.;
  }

  ux=gamma1*p->vx;
  uy=gamma1*p->vy;
  uz=gamma1*p->vz;

  gamma2=C/sqrt(C*C+ux*ux+uy*uy+uz*uz);

  bux=gamma2*(bx);
  buy=gamma2*(by);
  buz=gamma2*(bz);

  S=2.*T/(1.+T*T*(bux*bux+buy*buy+buz*buz));

  ux+=(ex)*T;
  uy+=(ey)*T;
  uz+=(ez)*T;

  uox=ux+T*(uy*buz-uz*buy);
  uoy=uy+T*(uz*bux-ux*buz);
  uoz=uz+T*(ux*buy-uy*bux);

  upx=ux+S*(uoy*buz-uoz*buy);
  upy=uy+S*(uoz*bux-uox*buz);
  upz=uz+S*(uox*buy-uoy*bux);

  upx+=(ex)*T;
  upy+=(ey)*T;
  upz+=(ez)*T;

  gamma2=C/sqrt(C*C+upx*upx+upy*upy+upz*upz);

  p->vx=gamma2*upx;
  p->vy=gamma2*upy;
  p->vz=gamma2*upz;

  p->x+=p->vx*dt*0.5;
  p->y+=p->vy*dt*0.5;
  p->z+=p->vz*dt*0.5;
  
  p->n=p->n;
  p->flag=p->flag;
  p->next_particle=p->next_particle;
  p->prev_particle=p->prev_particle;

  return(0);
}


inline int del_particle_i(Particle *ion[],const int i,Particle *ion_start[])
{
  Particle *p1;

  p1=ion[i]->next_particle;

  if(ion[i]->prev_particle==NULL){
    //ion_start[i]=ion[i]->next_particle;
    //free(ion[i]->next_particle->prev_particle);
    //ion[i]->next_particle->prev_particle=NULL;

    ion_start[i]=ion[i]->next_particle;
    ion[i]=ion_start[i];
    free(ion[i]->prev_particle);
    ion[i]->prev_particle=NULL;
  }else if(ion[i]->next_particle==NULL){
    ion[i]->prev_particle->next_particle=NULL;
    free(ion[i]);
  }else{
    ion[i]=ion[i]->prev_particle;
    free(ion[i]->next_particle);
    ion[i]->next_particle=p1;
    ion[i]->next_particle->prev_particle=ion[i];
  }

  return(0);
}

inline int del_particle_e(Particle *electron[],const int i,Particle *electron_start[])
{
  Particle *p1;

  p1=electron[i]->next_particle;

  if(electron[i]->prev_particle==NULL){
    //electron_start[i]=electron[i]->next_particle;
    //free(electron[i]->next_particle->prev_particle);
    //electron[i]->next_particle->prev_particle=NULL;

    electron_start[i]=electron[i]->next_particle;
    electron[i]=electron_start[i];
    free(electron[i]->prev_particle);
    electron[i]->prev_particle=NULL;
  }else if(electron[i]->next_particle==NULL){
    electron[i]->prev_particle->next_particle=NULL;
    free(electron[i]);
  }else{
    electron[i]=electron[i]->prev_particle;
    free(electron[i]->next_particle);
    electron[i]->next_particle=p1;
    electron[i]->next_particle->prev_particle=electron[i];
  }

  return(0);
}

/*******************************************************************
shape function
 grid->particle
******************************************************************/
inline int shape_func(const Particle *p,double *ex,double *ey,double *ez,double *bx,double *by,double *bz)
{
  int k=(int)(p->x/dx);
  int l=(int)(p->y/dx-0.5);
  int m=(int)(p->z/dx-0.5);
  int kkk=k+1;
  int lll=l+1;
  int mmm=m+1;

  double kkkp=(kkk*dx-p->x);
  double lllp=((lll+0.5)*dx-p->y);
  double mmmp=((mmm+0.5)*dx-p->z);
  double pk=(p->x-k*dx);
  double pl=(p->y-(l+0.5)*dx);
  double pm=(p->z-(m+0.5)*dx);

  *ex=(((grid[k][l][m].ex*kkkp)*(lllp*mmmp)+(grid[k][l][mmm].ex*kkkp)*(lllp*pm))+
       ((grid[k][lll][m].ex*kkkp)*(pl*mmmp)+(grid[k][lll][mmm].ex*kkkp)*(pl*pm))+
       ((grid[kkk][l][m].ex*pk)*(lllp*mmmp)+(grid[kkk][l][mmm].ex*pk)*(lllp*pm))+
       ((grid[kkk][lll][m].ex*pk)*(pl*mmmp)+(grid[kkk][lll][mmm].ex*pk)*(pl*pm)))/dx3;

  m=(int)(p->z/dx);
  mmm=m+1;
  mmmp=(mmm*dx-p->z);
  pm=(p->z-m*dx);

  *by=(((grid[k][l][m].by*kkkp)*(lllp*mmmp)+(grid[k][l][mmm].by*kkkp)*(lllp*pm))+
       ((grid[k][lll][m].by*kkkp)*(pl*mmmp)+(grid[k][lll][mmm].by*kkkp)*(pl*pm))+
       ((grid[kkk][l][m].by*pk)*(lllp*mmmp)+(grid[kkk][l][mmm].by*pk)*(lllp*pm))+
       ((grid[kkk][lll][m].by*pk)*(pl*mmmp)+(grid[kkk][lll][mmm].by*pk)*(pl*pm)))/dx3;

  k=(int)(p->x/dx-0.5);
  kkk=k+1;
  kkkp=((kkk+0.5)*dx-p->x);
  pk=(p->x-(k+0.5)*dx);

  *ez=(((grid[k][l][m].ez*kkkp)*(lllp*mmmp)+(grid[k][l][mmm].ez*kkkp)*(lllp*pm))+
       ((grid[k][lll][m].ez*kkkp)*(pl*mmmp)+(grid[k][lll][mmm].ez*kkkp)*(pl*pm))+
       ((grid[kkk][l][m].ez*pk)*(lllp*mmmp)+(grid[kkk][l][mmm].ez*pk)*(lllp*pm))+
       ((grid[kkk][lll][m].ez*pk)*(pl*mmmp)+(grid[kkk][lll][mmm].ez*pk)*(pl*pm)))/dx3;

  l=(int)(p->y/dx);
  lll=l+1;
  lllp=(lll*dx-p->y);
  pl=(p->y-l*dx);

  *bx=(((grid[k][l][m].bx*kkkp)*(lllp*mmmp)+(grid[k][l][mmm].bx*kkkp)*(lllp*pm))+
       ((grid[k][lll][m].bx*kkkp)*(pl*mmmp)+(grid[k][lll][mmm].bx*kkkp)*(pl*pm))+
       ((grid[kkk][l][m].bx*pk)*(lllp*mmmp)+(grid[kkk][l][mmm].bx*pk)*(lllp*pm))+
       ((grid[kkk][lll][m].bx*pk)*(pl*mmmp)+(grid[kkk][lll][mmm].bx*pk)*(pl*pm)))/dx3;


  m=(int)(p->z/dx-0.5);
  mmm=m+1;
  mmmp=((mmm+0.5)*dx-p->z);
  pm=(p->z-(m+0.5)*dx);

  *ey=(((grid[k][l][m].ey*kkkp)*(lllp*mmmp)+(grid[k][l][mmm].ey*kkkp)*(lllp*pm))+
       ((grid[k][lll][m].ey*kkkp)*(pl*mmmp)+(grid[k][lll][mmm].ey*kkkp)*(pl*pm))+
       ((grid[kkk][l][m].ey*pk)*(lllp*mmmp)+(grid[kkk][l][mmm].ey*pk)*(lllp*pm))+
       ((grid[kkk][lll][m].ey*pk)*(pl*mmmp)+(grid[kkk][lll][mmm].ey*pk)*(pl*pm)))/dx3;


  k=(int)(p->x/dx);
  kkk=k+1;
  kkkp=(kkk*dx-p->x);
  pk=(p->x-k*dx);

  *bz=(((grid[k][l][m].bz*kkkp)*(lllp*mmmp)+(grid[k][l][mmm].bz*kkkp)*(lllp*pm))+
       ((grid[k][lll][m].bz*kkkp)*(pl*mmmp)+(grid[k][lll][mmm].bz*kkkp)*(pl*pm))+
       ((grid[kkk][l][m].bz*pk)*(lllp*mmmp)+(grid[kkk][l][mmm].bz*pk)*(lllp*pm))+
       ((grid[kkk][lll][m].bz*pk)*(pl*mmmp)+(grid[kkk][lll][mmm].bz*pk)*(pl*pm)))/dx3;


  /**ex=0.;
  *ey=0.;
  *ez=0.;
  *bx=0.;
  *by=0.;
  *bz=0.;*/

  return(0);
}

inline int shape_func_v2(const Particle *p,double *ex,double *ey,double *ez,double *bx,double *by,double *bz)
{
  int k1,l1,m1,kkk1,lll1,mmm1;
  int k2,l2,m2,kkk2,lll2,mmm2;
  int k3,l3,m3,kkk3,lll3,mmm3;
  int k4,l4,m4,kkk4,lll4,mmm4;
  int k5,l5,m5,kkk5,lll5,mmm5;
  int k6,l6,m6,kkk6,lll6,mmm6;

  double kkkp1,lllp1,mmmp1;
  double kkkp2,lllp2,mmmp2;
  double kkkp3,lllp3,mmmp3;
  double kkkp4,lllp4,mmmp4;
  double kkkp5,lllp5,mmmp5;
  double kkkp6,lllp6,mmmp6;

  double pk1,pl1,pm1;
  double pk2,pl2,pm2;
  double pk3,pl3,pm3;
  double pk4,pl4,pm4;
  double pk5,pl5,pm5;
  double pk6,pl6,pm6;

  const double inv_dx3=1./(dx*dx*dx);

  double pex_1_1,pex_1_2,pex_1_3,pex_1_4,pex_2_1,pex_2_2,pex_2_3,pex_2_4;
  double pex_3_1,pex_3_2,pex_3_3,pex_3_4,pex_4_1,pex_4_2,pex_4_3,pex_4_4;

  double pby_1_1,pby_1_2,pby_1_3,pby_1_4,pby_2_1,pby_2_2,pby_2_3,pby_2_4;
  double pby_3_1,pby_3_2,pby_3_3,pby_3_4,pby_4_1,pby_4_2,pby_4_3,pby_4_4;

  double pez_1_1,pez_1_2,pez_1_3,pez_1_4,pez_2_1,pez_2_2,pez_2_3,pez_2_4;
  double pez_3_1,pez_3_2,pez_3_3,pez_3_4,pez_4_1,pez_4_2,pez_4_3,pez_4_4;

  double pbx_1_1,pbx_1_2,pbx_1_3,pbx_1_4,pbx_2_1,pbx_2_2,pbx_2_3,pbx_2_4;
  double pbx_3_1,pbx_3_2,pbx_3_3,pbx_3_4,pbx_4_1,pbx_4_2,pbx_4_3,pbx_4_4;

  double pey_1_1,pey_1_2,pey_1_3,pey_1_4,pey_2_1,pey_2_2,pey_2_3,pey_2_4;
  double pey_3_1,pey_3_2,pey_3_3,pey_3_4,pey_4_1,pey_4_2,pey_4_3,pey_4_4;

  double pbz_1_1,pbz_1_2,pbz_1_3,pbz_1_4,pbz_2_1,pbz_2_2,pbz_2_3,pbz_2_4;
  double pbz_3_1,pbz_3_2,pbz_3_3,pbz_3_4,pbz_4_1,pbz_4_2,pbz_4_3,pbz_4_4;

  /*
  k1 = k2      = k4 = k5      = (int)(p->x/dx);
  l1 = l2 = l3      = l5 = l6 = (int)(p->y/dx-0.5);
  m1      = m3 = m4      = m6 = (int)(p->z/dx-0.5);

  kkk1 = kkk2        = kkk4 = kkk5        = k1+1;
  lll1 = lll2 = lll3        = lll5 = lll6 = l1+1;
  mmm1        = mmm3 = mmm4        = mmm6 = m1+1;
  
  kkkp1 = kkkp2 =         kkkp4 = kkkp5         = (kkk1*dx-p->x);
  lllp1 = lllp2 = lllp3         = lllp5 = lllp6 = ((lll1+0.5)*dx-p->y);
  mmmp1         = mmmp3 = mmmp4         = mmmp6 = ((mmm1+0.5)*dx-p->z);
  
  pk1 = pk2       = pk4 = pk5       = (p->x-k1*dx);
  pl1 = pl2 = pl3       = pl5 = pl6 = (p->y-(l1+0.5)*dx);
  pm1       = pm3 = pm4       = pm6 = (p->z-(m1+0.5)*dx);
  
  m2=(int)(p->z/dx);
  mmm2=m2+1;
  mmmp2=(mmm2*dx-p->z);
  pm2=(p->z-m2*dx);
  
  k3=(int)(p->x/dx-0.5);
  kkk3=k3+1;
  kkkp3=((kkk3+0.5)*dx-p->x);
  pk3=(p->x-(k3+0.5)*dx);
  
  l4=(int)(p->y/dx);
  lll4=l4+1;
  lllp4=(lll4*dx-p->y);
  pl4=(p->y-l4*dx);
  
  m5=(int)(p->z/dx-0.5);
  mmm5=m5+1;
  mmmp5=((mmm5+0.5)*dx-p->z);
  pm5=(p->z-(m5+0.5)*dx);
  
  k6=(int)(p->x/dx);
  kkk6=k6+1;
  kkkp6=(kkk6*dx-p->x);
  pk6=(p->x-k6*dx);

*/

  k1 = k2 =                k6 = (int)(p->x/dx);
            k3 = k4 = k5      = (int)(p->x/dx-0.5);

	         l4 = l5 = l6 = (int)(p->y/dx);
  l1 = l2 = l3                = (int)(p->y/dx-0.5);
  
       m2 = m3 = m4           = (int)(p->z/dx);
  m1 =                m5 = m6 = (int)(p->z/dx-0.5);

  kkk1 = kkk2 =                      kkk6 = k1+1;
                kkk3 = kkk4 = kkk5        = k3+1;

                       lll4 = lll5 = lll6 = l4+1;
  lll1 = lll2 = lll3                      = l1+1;

         mmm2 = mmm3 = mmm4               = m2+1;
  mmm1 =                      mmm5 = mmm6 = m1+1;

  pk1 = pk2 =                   pk6 = (p->x-k1*dx);
              pk3 = pk4 = pk5       = (p->x-(k3+0.5)*dx);

	            pl4 = pl5 = pl6 = (p->y-l4*dx);
  pl1 = pl2 = pl3                   = (p->y-(l1+0.5)*dx);
  
        pm2 = pm3 = pm4             = (p->z-m2*dx);
  pm1 =                   pm5 = pm6 = (p->z-(m1+0.5)*dx);


  kkkp1 = kkkp2 =                         kkkp6 = (kkk1*dx-p->x);
                  kkkp3 = kkkp4 = kkkp5         = ((kkk3+0.5)*dx-p->x);

                          lllp4 = lllp5 = lllp6 = (lll4*dx-p->y);
  lllp1 = lllp2 = lllp3                         = ((lll1+0.5)*dx-p->y);

          mmmp2 = mmmp3 = mmmp4                 = (mmm2*dx-p->z);
  mmmp1 =                         mmmp5 = mmmp6 = ((mmm1+0.5)*dx-p->z);

  pex_1_1 = grid[k1][l1][m1].ex;////k1=(int)(p->x/dx),l1=(int)(p->y/dx-0.5),m1=(int)(p->z/dx-0.5)
  pex_1_2 = kkkp1*lllp1*mmmp1;
  pex_1_3 = grid[k1][l1][mmm1].ex;
  pex_1_4 = kkkp1*lllp1*pm1;
  
  pex_2_1 = grid[k1][lll1][m1].ex;
  pex_2_2 = kkkp1*pl1*mmmp1;
  pex_2_3 = grid[k1][lll1][mmm1].ex;
  pex_2_4 = kkkp1*pl1*pm1;
  
  pex_3_1 = grid[kkk1][l1][m1].ex;
  pex_3_2 = pk1*lllp1*mmmp1;
  pex_3_3 = grid[kkk1][l1][mmm1].ex;
  pex_3_4 = pk1*lllp1*pm1;
  
  pex_4_1 = grid[kkk1][lll1][m1].ex;
  pex_4_2 = pk1*pl1*mmmp1;
  pex_4_3 = grid[kkk1][lll1][mmm1].ex;
  pex_4_4 = pk1*pl1*pm1;
  
  pby_1_1 = grid[k2][l2][m2].by;////k2=(int)(p->x/dx),l2=(int)(p->y/dx-0.5),m2=(int)(p->z/dx)
  pby_1_2 = kkkp2*lllp2*mmmp2;
  pby_1_3 = grid[k2][l2][mmm2].by;
  pby_1_4 = kkkp2*lllp2*pm2;
  
  pby_2_1 = grid[k2][lll2][m2].by;
  pby_2_2 = kkkp2*pl2*mmmp2;
  pby_2_3 = grid[k2][lll2][mmm2].by;
  pby_2_4 = kkkp2*pl2*pm2;
  
  pby_3_1 = grid[kkk2][l2][m2].by;
  pby_3_2 = pk2*lllp2*mmmp2;
  pby_3_3 = grid[kkk2][l2][mmm2].by;
  pby_3_4 = pk2*lllp2*pm2;
  
  pby_4_1 = grid[kkk2][lll2][m2].by;
  pby_4_2 = pk2*pl2*mmmp2;
  pby_4_3 = grid[kkk2][lll2][mmm2].by;
  pby_4_4 = pk2*pl2*pm2;
  
  pez_1_1 = grid[k3][l3][m3].ez;////k3=(int)(p->x/dx-0.5),l3=(int)(p->y/dx-0.5),m3=(int)(p->z/dx)
  pez_1_2 = kkkp3*lllp3*mmmp3;
  pez_1_3 = grid[k3][l3][mmm3].ez;
  pez_1_4 = kkkp3*lllp3*pm3;
  
  pez_2_1 = grid[k3][lll3][m3].ez;
  pez_2_2 = kkkp3*pl3*mmmp3;
  pez_2_3 = grid[k3][lll3][mmm3].ez;
  pez_2_4 = kkkp3*pl3*pm3;
  
  pez_3_1 = grid[kkk3][l3][m3].ez;
  pez_3_2 = pk3*lllp3*mmmp3;
  pez_3_3 = grid[kkk3][l3][mmm3].ez;
  pez_3_4 = pk3*lllp3*pm3;
  
  pez_4_1 = grid[kkk3][lll3][m3].ez;
  pez_4_2 = pk3*pl3*mmmp3;
  pez_4_3 = grid[kkk3][lll3][mmm3].ez;
  pez_4_4 = pk3*pl3*pm3;
  
  pbx_1_1 = grid[k4][l4][m4].bx;////k4=(int)(p->x/dx-0.5),l4=(int)(p->y/dx),m4=(int)(p->z/dx)
  pbx_1_2 = kkkp4*lllp4*mmmp4;
  pbx_1_3 = grid[k4][l4][mmm4].bx;
  pbx_1_4 = kkkp4*lllp4*pm4;
  
  pbx_2_1 = grid[k4][lll4][m4].bx;
  pbx_2_2 = kkkp4*pl4*mmmp4;
  pbx_2_3 = grid[k4][lll4][mmm4].bx;
  pbx_2_4 = kkkp4*pl4*pm4;
  
  pbx_3_1 = grid[kkk4][l4][m4].bx;
  pbx_3_2 = pk4*lllp4*mmmp4;
  pbx_3_3 = grid[kkk4][l4][mmm4].bx;
  pbx_3_4 = pk4*lllp4*pm4;
  
  pbx_4_1 = grid[kkk4][lll4][m4].bx;
  pbx_4_2 = pk4*pl4*mmmp4;
  pbx_4_3 = grid[kkk4][lll4][mmm4].bx;
  pbx_4_4 = pk4*pl4*pm4;
  
  pey_1_1 = grid[k5][l5][m5].ey;////k5=(int)(p->x/dx-0.5),l5=(int)(p->y/dx),m5=(int)(p->z/dx-0.5)
  pey_1_2 = kkkp5*lllp5*mmmp5;
  pey_1_3 = grid[k5][l5][mmm5].ey;
  pey_1_4 = kkkp5*lllp5*pm5;
  
  pey_2_1 = grid[k5][lll5][m5].ey;
  pey_2_2 = kkkp5*pl5*mmmp5;
  pey_2_3 = grid[k5][lll5][mmm5].ey;
  pey_2_4 = kkkp5*pl5*pm5;
  
  pey_3_1 = grid[kkk5][l5][m5].ey;
  pey_3_2 = pk5*lllp5*mmmp5;
  pey_3_3 = grid[kkk5][l5][mmm5].ey;
  pey_3_4 = pk5*lllp5*pm5;
  
  pey_4_1 = grid[kkk5][lll5][m5].ey;
  pey_4_2 = pk5*pl5*mmmp5;
  pey_4_3 = grid[kkk5][lll5][mmm5].ey;
  pey_4_4 = pk5*pl5*pm5;
  
  pbz_1_1 = grid[k6][l6][m6].bz;////k6=(int)(p->x/dx),l6=(int)(p->y/dx),m6=(int)(p->z/dx-0.5)
  pbz_1_2 = kkkp6*lllp6*mmmp6;
  pbz_1_3 = grid[k6][l6][mmm6].bz;
  pbz_1_4 = kkkp6*lllp6*pm6;
  
  pbz_2_1 = grid[k6][lll6][m6].bz;
  pbz_2_2 = kkkp6*pl6*mmmp6;
  pbz_2_3 = grid[k6][lll6][mmm6].bz;
  pbz_2_4 = kkkp6*pl6*pm6;
  
  pbz_3_1 = grid[kkk6][l6][m6].bz;
  pbz_3_2 = pk6*lllp6*mmmp6;
  pbz_3_3 = grid[kkk6][l6][mmm6].bz;
  pbz_3_4 = pk6*lllp6*pm6;
  
  pbz_4_1 = grid[kkk6][lll6][m6].bz;
  pbz_4_2 = pk6*pl6*mmmp6;
  pbz_4_3 = grid[kkk6][lll6][mmm6].bz;
  pbz_4_4 = pk6*pl6*pm6;
  
  
  pex_1_1 = pex_1_1 * pex_1_2 + pex_1_3 * pex_1_4;
  pby_1_1 = pby_1_1 * pby_1_2 + pby_1_3 * pby_1_4;
  pez_1_1 = pez_1_1 * pez_1_2 + pez_1_3 * pez_1_4;
  pbx_1_1 = pbx_1_1 * pbx_1_2 + pbx_1_3 * pbx_1_4;
  pey_1_1 = pey_1_1 * pey_1_2 + pey_1_3 * pey_1_4;
  pbz_1_1 = pbz_1_1 * pbz_1_2 + pbz_1_3 * pbz_1_4;
  
  pex_2_1 = pex_2_1 * pex_2_2 + pex_2_3 * pex_2_4;
  pby_2_1 = pby_2_1 * pby_2_2 + pby_2_3 * pby_2_4;
  pez_2_1 = pez_2_1 * pez_2_2 + pez_2_3 * pez_2_4;
  pbx_2_1 = pbx_2_1 * pbx_2_2 + pbx_2_3 * pbx_2_4;
  pey_2_1 = pey_2_1 * pey_2_2 + pey_2_3 * pey_2_4;
  pbz_2_1 = pbz_2_1 * pbz_2_2 + pbz_2_3 * pbz_2_4;
  
  pex_3_1 = pex_3_1 * pex_3_2 + pex_3_3 * pex_3_4;
  pby_3_1 = pby_3_1 * pby_3_2 + pby_3_3 * pby_3_4;
  pez_3_1 = pez_3_1 * pez_3_2 + pez_3_3 * pez_3_4;
  pbx_3_1 = pbx_3_1 * pbx_3_2 + pbx_3_3 * pbx_3_4;
  pey_3_1 = pey_3_1 * pey_3_2 + pey_3_3 * pey_3_4;
  pbz_3_1 = pbz_3_1 * pbz_3_2 + pbz_3_3 * pbz_3_4;
  
  pex_4_1 = pex_4_1 * pex_4_2 + pex_4_3 * pex_4_4;
  pby_4_1 = pby_4_1 * pby_4_2 + pby_4_3 * pby_4_4;
  pez_4_1 = pez_4_1 * pez_4_2 + pez_4_3 * pez_4_4;
  pbx_4_1 = pbx_4_1 * pbx_4_2 + pbx_4_3 * pbx_4_4;
  pey_4_1 = pey_4_1 * pey_4_2 + pey_4_3 * pey_4_4;
  pbz_4_1 = pbz_4_1 * pbz_4_2 + pbz_4_3 * pbz_4_4;
  
  *ex = (pex_1_1 + pex_2_1 + pex_3_1 + pex_4_1)*inv_dx3;
  *by = (pby_1_1 + pby_2_1 + pby_3_1 + pby_4_1)*inv_dx3;
  *ez = (pez_1_1 + pez_2_1 + pez_3_1 + pez_4_1)*inv_dx3;
  *bx = (pbx_1_1 + pbx_2_1 + pbx_3_1 + pbx_4_1)*inv_dx3;
  *ey = (pey_1_1 + pey_2_1 + pey_3_1 + pey_4_1)*inv_dx3;
  *bz = (pbz_1_1 + pbz_2_1 + pbz_3_1 + pbz_4_1)*inv_dx3;

    /* p->ex=(((grid[k1][l1][m1].ex*kkkp1)*(lllp1*mmmp1)+(grid[k1][l1][mmm1].ex*kkkp1)*(lllp1*pm1))+
	   ((grid[k1][lll1][m1].ex*kkkp1)*(pl1*mmmp1)+(grid[k1][lll1][mmm1].ex*kkkp1)*(pl1*pm1))+
	   ((grid[kkk1][l1][m1].ex*pk1)*(lllp1*mmmp1)+(grid[kkk1][l1][mmm1].ex*pk1)*(lllp1*pm1))+
	   ((grid[kkk1][lll1][m1].ex*pk1)*(pl1*mmmp1)+(grid[kkk1][lll1][mmm1].ex*pk1)*(pl1*pm1)))*inv_dx3; */
    
    /* p->by=(((grid[k2][l2][m2].by*kkkp2)*(lllp2*mmmp2)+(grid[k2][l2][mmm2].by*kkkp2)*(lllp2*pm2))+
	   ((grid[k2][lll2][m2].by*kkkp2)*(pl2*mmmp2)+(grid[k2][lll2][mmm2].by*kkkp2)*(pl2*pm2))+
	   ((grid[kkk2][l2][m2].by*pk2)*(lllp2*mmmp2)+(grid[kkk2][l2][mmm2].by*pk2)*(lllp2*pm2))+
	   ((grid[kkk2][lll2][m2].by*pk2)*(pl2*mmmp2)+(grid[kkk2][lll2][mmm2].by*pk2)*(pl2*pm2)))*inv_dx3; */
    
    /* p->ez=(((grid[k3][l3][m3].ez*kkkp3)*(lllp3*mmmp3)+(grid[k3][l3][mmm3].ez*kkkp3)*(lllp3*pm3))+
	   ((grid[k3][lll3][m3].ez*kkkp3)*(pl3*mmmp3)+(grid[k3][lll3][mmm3].ez*kkkp3)*(pl3*pm3))+
	   ((grid[kkk3][l3][m3].ez*pk3)*(lllp3*mmmp3)+(grid[kkk3][l3][mmm3].ez*pk3)*(lllp3*pm3))+
	   ((grid[kkk3][lll3][m3].ez*pk3)*(pl3*mmmp3)+(grid[kkk3][lll3][mmm3].ez*pk3)*(pl3*pm3)))*inv_dx3; */
    
    /* p->bx=(((grid[k4][l4][m4].bx*kkkp4)*(lllp4*mmmp4)+(grid[k4][l4][mmm4].bx*kkkp4)*(lllp4*pm4))+
	   ((grid[k4][lll4][m4].bx*kkkp4)*(pl4*mmmp4)+(grid[k4][lll4][mmm4].bx*kkkp4)*(pl4*pm4))+
	   ((grid[kkk4][l4][m4].bx*pk4)*(lllp4*mmmp4)+(grid[kkk4][l4][mmm4].bx*pk4)*(lllp4*pm4))+
	   ((grid[kkk4][lll4][m4].bx*pk4)*(pl4*mmmp4)+(grid[kkk4][lll4][mmm4].bx*pk4)*(pl4*pm4)))*inv_dx3; */

    /* p->ey=(((grid[k5][l5][m5].ey*kkkp5)*(lllp5*mmmp5)+(grid[k5][l5][mmm5].ey*kkkp5)*(lllp5*pm5))+
	   ((grid[k5][lll5][m5].ey*kkkp5)*(pl5*mmmp5)+(grid[k5][lll5][mmm5].ey*kkkp5)*(pl5*pm5))+
	   ((grid[kkk5][l5][m5].ey*pk5)*(lllp5*mmmp5)+(grid[kkk5][l5][mmm5].ey*pk5)*(lllp5*pm5))+
	   ((grid[kkk5][lll5][m5].ey*pk5)*(pl5*mmmp5)+(grid[kkk5][lll5][mmm5].ey*pk5)*(pl5*pm5)))*inv_dx3; */
    
    /* p->bz=(((grid[k6][l6][m6].bz*kkkp6)*(lllp6*mmmp6)+(grid[k6][l6][mmm6].bz*kkkp6)*(lllp6*pm6))+
	   ((grid[k6][lll6][m6].bz*kkkp6)*(pl6*mmmp6)+(grid[k6][lll6][mmm6].bz*kkkp6)*(pl6*pm6))+
	   ((grid[kkk6][l6][m6].bz*pk6)*(lllp6*mmmp6)+(grid[kkk6][l6][mmm6].bz*pk6)*(lllp6*pm6))+
	   ((grid[kkk6][lll6][m6].bz*pk6)*(pl6*mmmp6)+(grid[kkk6][lll6][mmm6].bz*pk6)*(pl6*pm6)))*inv_dx3; */
}

/*******************************************************************
shape function 2
 particle->grid
******************************************************************/

inline int shape_func_ion0_n_2(const Particle *p,const int thread)
{
  const int k=(int)(p->x/dx-0.5);
  const int l=(int)(p->y/dx-0.5);
  const int m=(int)(p->z/dx-0.5);
  const int kkk=k+1;
  const int lll=l+1;
  const int mmm=m+1;
  const double temp=p->n/dx3;

  const double kkkp=((kkk+0.5)*dx-p->x);
  const double lllp=((lll+0.5)*dx-p->y);
  const double mmmp=((mmm+0.5)*dx-p->z);
  const double pk=(p->x-(k+0.5)*dx);
  const double pl=(p->y-(l+0.5)*dx);
  const double pm=(p->z-(m+0.5)*dx);

  const double tempkkkp=temp*kkkp;
  const double temppk=temp*pk; 

  grid_thread[thread][k][l][m].ni+=(tempkkkp)*(lllp*mmmp);
  grid_thread[thread][k][l][mmm].ni+=(tempkkkp)*(lllp*pm);
  grid_thread[thread][k][lll][m].ni+=(tempkkkp)*(pl*mmmp);
  grid_thread[thread][k][lll][mmm].ni+=(tempkkkp)*(pl*pm);
  grid_thread[thread][kkk][l][m].ni+=(temppk)*(lllp*mmmp);
  grid_thread[thread][kkk][l][mmm].ni+=(temppk)*(lllp*pm);
  grid_thread[thread][kkk][lll][m].ni+=(temppk)*(pl*mmmp);
  grid_thread[thread][kkk][lll][mmm].ni+=(temppk)*(pl*pm);

  return(0);
}

inline int shape_func_electron0_n_2(const Particle *p,const int thread)
{
  const int k=(int)(p->x/dx-0.5);
  const int l=(int)(p->y/dx-0.5);
  const int m=(int)(p->z/dx-0.5);
  const int kkk=k+1;
  const int lll=l+1;
  const int mmm=m+1;
  const double temp=p->n/dx3;

  const double kkkp=((kkk+0.5)*dx-p->x);
  const double lllp=((lll+0.5)*dx-p->y);
  const double mmmp=((mmm+0.5)*dx-p->z);
  const double pk=(p->x-(k+0.5)*dx);
  const double pl=(p->y-(l+0.5)*dx);
  const double pm=(p->z-(m+0.5)*dx);

  const double tempkkkp=temp*kkkp;
  const double temppk=temp*pk; 

  grid_thread[thread][k][l][m].ne+=(tempkkkp)*(lllp*mmmp);
  grid_thread[thread][k][l][mmm].ne+=(tempkkkp)*(lllp*pm);
  grid_thread[thread][k][lll][m].ne+=(tempkkkp)*(pl*mmmp);
  grid_thread[thread][k][lll][mmm].ne+=(tempkkkp)*(pl*pm);
  grid_thread[thread][kkk][l][m].ne+=(temppk)*(lllp*mmmp);
  grid_thread[thread][kkk][l][mmm].ne+=(temppk)*(lllp*pm);
  grid_thread[thread][kkk][lll][m].ne+=(temppk)*(pl*mmmp);
  grid_thread[thread][kkk][lll][mmm].ne+=(temppk)*(pl*pm);

  return(0);
}

inline int shape_func_ion0_j_2(const Particle *p,const int thread)
{
  int k=(int)(p->x/dx);
  int l=(int)(p->y/dx-0.5);
  int m=(int)(p->z/dx-0.5);
  int kkk=k+1;
  int lll=l+1;
  int mmm=m+1;

  const double qn=q*p->n/dx3;
  double temp=qn*p->vx;
  double kkkp=(kkk*dx-p->x);
  double lllp=((lll+0.5)*dx-p->y);
  double mmmp=((mmm+0.5)*dx-p->z);
  double pk=(p->x-k*dx);
  double pl=(p->y-(l+0.5)*dx);
  double pm=(p->z-(m+0.5)*dx);

  double tempkkkp=temp*kkkp;
  double temppk=temp*pk; 

  grid_thread[thread][k][l][m].jix0+=(tempkkkp)*(lllp*mmmp);
  grid_thread[thread][k][l][mmm].jix0+=(tempkkkp)*(lllp*pm);
  grid_thread[thread][k][lll][m].jix0+=(tempkkkp)*(pl*mmmp);
  grid_thread[thread][k][lll][mmm].jix0+=(tempkkkp)*(pl*pm);
  grid_thread[thread][kkk][l][m].jix0+=(temppk)*(lllp*mmmp);
  grid_thread[thread][kkk][l][mmm].jix0+=(temppk)*(lllp*pm);
  grid_thread[thread][kkk][lll][m].jix0+=(temppk)*(pl*mmmp);
  grid_thread[thread][kkk][lll][mmm].jix0+=(temppk)*(pl*pm);

  k=(int)(p->x/dx-0.5);
  l=(int)(p->y/dx);
  kkk=k+1;
  lll=l+1;
  temp=qn*p->vy;

  kkkp=((kkk+0.5)*dx-p->x);
  lllp=(lll*dx-p->y);
  pk=(p->x-(k+0.5)*dx);
  pl=(p->y-l*dx);

  tempkkkp=temp*kkkp;
  temppk=temp*pk; 

  grid_thread[thread][k][l][m].jiy0+=(tempkkkp)*(lllp*mmmp);
  grid_thread[thread][k][l][mmm].jiy0+=(tempkkkp)*(lllp*pm);
  grid_thread[thread][k][lll][m].jiy0+=(tempkkkp)*(pl*mmmp);
  grid_thread[thread][k][lll][mmm].jiy0+=(tempkkkp)*(pl*pm);
  grid_thread[thread][kkk][l][m].jiy0+=(temppk)*(lllp*mmmp);
  grid_thread[thread][kkk][l][mmm].jiy0+=(temppk)*(lllp*pm);
  grid_thread[thread][kkk][lll][m].jiy0+=(temppk)*(pl*mmmp);
  grid_thread[thread][kkk][lll][mmm].jiy0+=(temppk)*(pl*pm);

  l=(int)(p->y/dx-0.5);
  m=(int)(p->z/dx);
  lll=l+1;
  mmm=m+1;
  temp=qn*p->vz;

  lllp=((lll+0.5)*dx-p->y);
  mmmp=(mmm*dx-p->z);
  pl=(p->y-(l+0.5)*dx);
  pm=(p->z-m*dx);

  tempkkkp=temp*kkkp;
  temppk=temp*pk; 

  grid_thread[thread][k][l][m].jiz0+=(tempkkkp)*(lllp*mmmp);
  grid_thread[thread][k][l][mmm].jiz0+=(tempkkkp)*(lllp*pm);
  grid_thread[thread][k][lll][m].jiz0+=(tempkkkp)*(pl*mmmp);
  grid_thread[thread][k][lll][mmm].jiz0+=(tempkkkp)*(pl*pm);
  grid_thread[thread][kkk][l][m].jiz0+=(temppk)*(lllp*mmmp);
  grid_thread[thread][kkk][l][mmm].jiz0+=(temppk)*(lllp*pm);
  grid_thread[thread][kkk][lll][m].jiz0+=(temppk)*(pl*mmmp);
  grid_thread[thread][kkk][lll][mmm].jiz0+=(temppk)*(pl*pm);

  return(0);
}

inline int shape_func_electron0_j_2(const Particle *p,const int thread)
{
  int k=(int)(p->x/dx);
  int l=(int)(p->y/dx-0.5);
  int m=(int)(p->z/dx-0.5);
  int kkk=k+1;
  int lll=l+1;
  int mmm=m+1;

  const double qn=q*p->n/dx3;
  double temp=qn*p->vx;
  double kkkp=(kkk*dx-p->x);
  double lllp=((lll+0.5)*dx-p->y);
  double mmmp=((mmm+0.5)*dx-p->z);
  double pk=(p->x-k*dx);
  double pl=(p->y-(l+0.5)*dx);
  double pm=(p->z-(m+0.5)*dx);

  double tempkkkp=temp*kkkp;
  double temppk=temp*pk; 

  grid_thread[thread][k][l][m].jex0+=(tempkkkp)*(lllp*mmmp);
  grid_thread[thread][k][l][mmm].jex0+=(tempkkkp)*(lllp*pm);
  grid_thread[thread][k][lll][m].jex0+=(tempkkkp)*(pl*mmmp);
  grid_thread[thread][k][lll][mmm].jex0+=(tempkkkp)*(pl*pm);
  grid_thread[thread][kkk][l][m].jex0+=(temppk)*(lllp*mmmp);
  grid_thread[thread][kkk][l][mmm].jex0+=(temppk)*(lllp*pm);
  grid_thread[thread][kkk][lll][m].jex0+=(temppk)*(pl*mmmp);
  grid_thread[thread][kkk][lll][mmm].jex0+=(temppk)*(pl*pm);

  k=(int)(p->x/dx-0.5);
  l=(int)(p->y/dx);
  kkk=k+1;
  lll=l+1;
  temp=qn*p->vy;

  kkkp=((kkk+0.5)*dx-p->x);
  lllp=(lll*dx-p->y);
  pk=(p->x-(k+0.5)*dx);
  pl=(p->y-l*dx);

  tempkkkp=temp*kkkp;
  temppk=temp*pk; 

  grid_thread[thread][k][l][m].jey0+=(tempkkkp)*(lllp*mmmp);
  grid_thread[thread][k][l][mmm].jey0+=(tempkkkp)*(lllp*pm);
  grid_thread[thread][k][lll][m].jey0+=(tempkkkp)*(pl*mmmp);
  grid_thread[thread][k][lll][mmm].jey0+=(tempkkkp)*(pl*pm);
  grid_thread[thread][kkk][l][m].jey0+=(temppk)*(lllp*mmmp);
  grid_thread[thread][kkk][l][mmm].jey0+=(temppk)*(lllp*pm);
  grid_thread[thread][kkk][lll][m].jey0+=(temppk)*(pl*mmmp);
  grid_thread[thread][kkk][lll][mmm].jey0+=(temppk)*(pl*pm);

  l=(int)(p->y/dx-0.5);
  m=(int)(p->z/dx);
  lll=l+1;
  mmm=m+1;
  temp=qn*p->vz;

  lllp=((lll+0.5)*dx-p->y);
  mmmp=(mmm*dx-p->z);
  pl=(p->y-(l+0.5)*dx);
  pm=(p->z-m*dx);

  tempkkkp=temp*kkkp;
  temppk=temp*pk; 

  grid_thread[thread][k][l][m].jez0+=(tempkkkp)*(lllp*mmmp);
  grid_thread[thread][k][l][mmm].jez0+=(tempkkkp)*(lllp*pm);
  grid_thread[thread][k][lll][m].jez0+=(tempkkkp)*(pl*mmmp);
  grid_thread[thread][k][lll][mmm].jez0+=(tempkkkp)*(pl*pm);
  grid_thread[thread][kkk][l][m].jez0+=(temppk)*(lllp*mmmp);
  grid_thread[thread][kkk][l][mmm].jez0+=(temppk)*(lllp*pm);
  grid_thread[thread][kkk][lll][m].jez0+=(temppk)*(pl*mmmp);
  grid_thread[thread][kkk][lll][mmm].jez0+=(temppk)*(pl*pm);

  return(0);
}

int external_current(const int myrank_x,const int myrank_y,const int c)
{
  const double w=Pi/50000.;
  int i,k,l,m;
  double L=4*dx;//=R_coil
  double x0,y0,z0,x1,y1,z1,r1;
  double J0=26.06;//424.4105;//L=4*dx
  double t0=0.,p0=0.;
  double jx1,jy1,jz1;
  // quad2d(@(r,y) J0*power(r/L,2).*exp(-power(r/L,2)).*exp(-power(y/L.*2,2))*pi.*r.*r,0,10000,-1000,1000)=pi*R*R*I

  if(c<50000){
    for(k=2;k<Grid_Nx+2;k++){
      for(l=2;l<Grid_Ny+2;l++){
	for(m=2;m<Grid_Nz+2;m++){
	  x0=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y0=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  z0=(m-(Grid_Nz+4)/2+0.5)*dx;

	  x1=x0*cos(p0)*cos(t0)+y0*sin(p0)*cos(t0)-z0*sin(t0);
	  y1=-x0*sin(p0)+y0*cos(p0);
	  z1=x0*cos(p0)*sin(t0)+y0*sin(p0)*sin(t0)+z0*cos(t0);
	  r1=sqrt(x1*x1+y1*y1);

	  jx1=J0*(1.+cos(w*c-Pi))/2.*pow(r1/L,2)*exp(-pow(r1/L,2))*exp(-pow(z1/L*2,2))*(-y1/r1);
	  jy1=J0*(1.+cos(w*c-Pi))/2.*pow(r1/L,2)*exp(-pow(r1/L,2))*exp(-pow(z1/L*2,2))*(x1/r1);
	  jz1=0.;

	  grid[k][l][m].jix0+=jx1*cos(p0)*cos(t0)-jy1*sin(p0)+jz1*cos(p0)*sin(t0);

	  x0=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  y0=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z0=(m-(Grid_Nz+4)/2+0.5)*dx;

	  x1=x0*cos(p0)*cos(t0)+y0*sin(p0)*cos(t0)-z0*sin(t0);
	  y1=-x0*sin(p0)+y0*cos(p0);
	  z1=x0*cos(p0)*sin(t0)+y0*sin(p0)*sin(t0)+z0*cos(t0);
	  r1=sqrt(x1*x1+y1*y1);

	  jx1=J0*(1.+cos(w*c-Pi))/2.*pow(r1/L,2)*exp(-pow(r1/L,2))*exp(-pow(z1/L*2,2))*(-y1/r1);
	  jy1=J0*(1.+cos(w*c-Pi))/2.*pow(r1/L,2)*exp(-pow(r1/L,2))*exp(-pow(z1/L*2,2))*(x1/r1);
	  jz1=0.;

	  grid[k][l][m].jiy0+=jx1*sin(p0)*cos(t0)+jy1*cos(p0)+jz1*sin(p0)*sin(t0);

	  x0=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  y0=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  z0=(m-(Grid_Nz+4)/2)*dx;

	  x1=x0*cos(p0)*cos(t0)+y0*sin(p0)*cos(t0)-z0*sin(t0);
	  y1=-x0*sin(p0)+y0*cos(p0);
	  z1=x0*cos(p0)*sin(t0)+y0*sin(p0)*sin(t0)+z0*cos(t0);
	  r1=sqrt(x1*x1+y1*y1);

	  jx1=J0*(1.+cos(w*c-Pi))/2.*pow(r1/L,2)*exp(-pow(r1/L,2))*exp(-pow(z1/L*2,2))*(-y1/r1);
	  jy1=J0*(1.+cos(w*c-Pi))/2.*pow(r1/L,2)*exp(-pow(r1/L,2))*exp(-pow(z1/L*2,2))*(x1/r1);
	  jz1=0.;

	  grid[k][l][m].jiz0+=-jx1*sin(t0)+jz1*cos(t0);
	}
      }
    }
  }else{
    for(k=2;k<Grid_Nx+2;k++){
      for(l=2;l<Grid_Ny+2;l++){
	for(m=2;m<Grid_Nz+2;m++){
	  x0=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y0=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  z0=(m-(Grid_Nz+4)/2+0.5)*dx;

	  x1=x0*cos(p0)*cos(t0)+y0*sin(p0)*cos(t0)-z0*sin(t0);
	  y1=-x0*sin(p0)+y0*cos(p0);
	  z1=x0*cos(p0)*sin(t0)+y0*sin(p0)*sin(t0)+z0*cos(t0);
	  r1=sqrt(x1*x1+y1*y1);

	  jx1=J0*pow(r1/L,2)*exp(-pow(r1/L,2))*exp(-pow(z1/L*2,2))*(-y1/r1);
	  jy1=J0*pow(r1/L,2)*exp(-pow(r1/L,2))*exp(-pow(z1/L*2,2))*(x1/r1);
	  jz1=0.;

	  grid[k][l][m].jix0+=jx1*cos(p0)*cos(t0)-jy1*sin(p0)+jz1*cos(p0)*sin(t0);

	  x0=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  y0=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z0=(m-(Grid_Nz+4)/2+0.5)*dx;

	  x1=x0*cos(p0)*cos(t0)+y0*sin(p0)*cos(t0)-z0*sin(t0);
	  y1=-x0*sin(p0)+y0*cos(p0);
	  z1=x0*cos(p0)*sin(t0)+y0*sin(p0)*sin(t0)+z0*cos(t0);
	  r1=sqrt(x1*x1+y1*y1);

	  jx1=J0*pow(r1/L,2)*exp(-pow(r1/L,2))*exp(-pow(z1/L*2,2))*(-y1/r1);
	  jy1=J0*pow(r1/L,2)*exp(-pow(r1/L,2))*exp(-pow(z1/L*2,2))*(x1/r1);
	  jz1=0.;

	  grid[k][l][m].jiy0+=jx1*sin(p0)*cos(t0)+jy1*cos(p0)+jz1*sin(p0)*sin(t0);

	  x0=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  y0=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  z0=(m-(Grid_Nz+4)/2)*dx;

	  x1=x0*cos(p0)*cos(t0)+y0*sin(p0)*cos(t0)-z0*sin(t0);
	  y1=-x0*sin(p0)+y0*cos(p0);
	  z1=x0*cos(p0)*sin(t0)+y0*sin(p0)*sin(t0)+z0*cos(t0);
	  r1=sqrt(x1*x1+y1*y1);

	  jx1=J0*pow(r1/L,2)*exp(-pow(r1/L,2))*exp(-pow(z1/L*2,2))*(-y1/r1);
	  jy1=J0*pow(r1/L,2)*exp(-pow(r1/L,2))*exp(-pow(z1/L*2,2))*(x1/r1);
	  jz1=0.;

	  grid[k][l][m].jiz0+=-jx1*sin(t0)+jz1*cos(t0);
	}
      }
    }
  }

  return(0);
}

int external_current_para(const int myrank_x,const int myrank_y,const int c)
{
  const double w=Pi/50000.;
  int i,k,l,m;
  double L=4*dx;//R_coil~8*dx
  double xx,yy,zz,r;
  double I0=I/3.1416;//L=4*dx

  if(c<50000){
    for(k=2;k<Grid_Nx+2;k++){
      for(l=2;l<Grid_Ny+2;l++){
	for(m=2;m<Grid_Nz+2;m++){
	  xx=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  yy=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  zz=(m-(Grid_Nz+4)/2+0.5)*dx;
	  r=sqrt(xx*xx+yy*yy);
	  
	  grid[k][l][m].jix0+=I0*(1.+cos(w*c-Pi))/2.*pow(r/L,2)*exp(-pow(r/L,2))*exp(-pow(zz/L*2,2))*(-yy/r);

	  xx=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  yy=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  zz=(m-(Grid_Nz+4)/2+0.5)*dx;
	  r=sqrt(xx*xx+yy*yy);

	  grid[k][l][m].jiy0+=I0*(1.+cos(w*c-Pi))/2.*pow(r/L,2)*exp(-pow(r/L,2))*exp(-pow(zz/L*2,2))*(xx/r);
	}
      }
    }
  }else{
    for(k=2;k<Grid_Nx+2;k++){
      for(l=2;l<Grid_Ny+2;l++){
	for(m=2;m<Grid_Nz+2;m++){
	  xx=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  yy=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  zz=(m-(Grid_Nz+4)/2+0.5)*dx;
	  r=sqrt(xx*xx+yy*yy);
	  
	  grid[k][l][m].jix0+=I0*pow(r/L,2)*exp(-pow(r/L,2))*exp(-pow(zz/L*2,2))*(-yy/r);

	  xx=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  yy=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  zz=(m-(Grid_Nz+4)/2+0.5)*dx;
	  r=sqrt(xx*xx+yy*yy);

	  grid[k][l][m].jiy0+=I0*pow(r/L,2)*exp(-pow(r/L,2))*exp(-pow(zz/L*2,2))*(xx/r);
	}
      }
    }
  }

  return(0);
}

int external_current_perp(const int myrank_x,const int myrank_y,const int c)
{
  const double w=Pi/50000.;
  int i,k,l,m;
  double L=4*dx;//R_coil~8*dx
  double xx,yy,zz,r;
  double I0=I/3.1416;//L=4*dx


  if(c<50000){
    for(k=2;k<Grid_Nx+2;k++){
      for(l=2;l<Grid_Ny+2;l++){
	for(m=2;m<Grid_Nz+2;m++){
	  xx=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  yy=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  zz=(m-(Grid_Nz+4)/2+0.5)*dx;
	  r=sqrt(xx*xx+zz*zz);
	  
	  grid[k][l][m].jix0+=I0*(1.+cos(w*c-Pi))/2.*pow(r/L,2)*exp(-pow(r/L,2))*exp(-pow(yy/L*2,2))*(-zz/r);

	  xx=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  yy=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  zz=(m-(Grid_Nz+4)/2)*dx;
	  r=sqrt(xx*xx+zz*zz);

	  grid[k][l][m].jiz0+=I0*(1.+cos(w*c-Pi))/2.*pow(r/L,2)*exp(-pow(r/L,2))*exp(-pow(yy/L*2,2))*(xx/r);
	}
      }
    }
  }else{
    for(k=2;k<Grid_Nx+2;k++){
      for(l=2;l<Grid_Ny+2;l++){
	for(m=2;m<Grid_Nz+2;m++){
	  xx=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  yy=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  zz=(m-(Grid_Nz+4)/2+0.5)*dx;
	  r=sqrt(xx*xx+zz*zz);
	  
	  grid[k][l][m].jix0+=I0*pow(r/L,2)*exp(-pow(r/L,2))*exp(-pow(yy/L*2,2))*(-zz/r);

	  xx=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  yy=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  zz=(m-(Grid_Nz+4)/2)*dx;
	  r=sqrt(xx*xx+zz*zz);

	  grid[k][l][m].jiz0+=I0*pow(r/L,2)*exp(-pow(r/L,2))*exp(-pow(yy/L*2,2))*(xx/r);
	}
      }
    }
  }

  return(0);
}

int external_current1(const int myrank_x,const int myrank_y,const int c)
{
  const double w=Pi/50000.;
  int i,k,l,m;
  double x,y,z,r;
  double xs,ys,zs,rs;

  if(c<50000){
#pragma omp parallel for private(l,m,x,y,z,r)
    for(k=0;k<Grid_Nx+4;k++){
      for(l=0;l<Grid_Ny+4;l++){
	for(m=0;m<Grid_Nz+4;m++){
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z=(m-(Grid_Nz+4)/2)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);
	  
	  grid[k][l][m].bx-=grid[k][l][m].b0x;
	  grid[k][l][m].b0x=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5))+IMF_x;
	  grid[k][l][m].bx+=grid[k][l][m].b0x;
	  
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  z=(m-(Grid_Nz+4)/2)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);
	  
	  grid[k][l][m].by-=grid[k][l][m].b0y;
	  grid[k][l][m].b0y=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5))+IMF_y;
	  grid[k][l][m].by+=grid[k][l][m].b0y;

	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);

	  grid[k][l][m].bz-=grid[k][l][m].b0z;
	  grid[k][l][m].b0z=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5))+IMF_z;
	  grid[k][l][m].bz+=grid[k][l][m].b0z;
	}
      }
    }
  }else if (c>=50000&&c<100000){
#pragma omp parallel for private(l,m,x,y,z,r)
    for(k=0;k<Grid_Nx+4;k++){
      for(l=0;l<Grid_Ny+4;l++){
	for(m=0;m<Grid_Nz+4;m++){
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z=(m-(Grid_Nz+4)/2)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);
	  
	  grid[k][l][m].b0x=mu0*I*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5))+IMF_x;
	  
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  z=(m-(Grid_Nz+4)/2)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);
	  
	  grid[k][l][m].b0y=mu0*I*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5))+IMF_y;
	  
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);
	  
	  grid[k][l][m].b0z=mu0*I*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5))+IMF_z;
	}
      }
    }
  }else if(c>=100000&&c<150000){
#pragma omp parallel for private(l,m,x,y,z,r,xs,ys,zs,rs)
    for(k=0;k<Grid_Nx+4;k++){
      for(l=0;l<Grid_Ny+4;l++){
	for(m=0;m<Grid_Nz+4;m++){
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z=(m-(Grid_Nz+4)/2)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);

	  xs=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  ys=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx+R;
	  zs=(m-(Grid_Nz+4)/2)*dx;
	  
	  rs=sqrt(xs*xs+ys*ys+zs*zs);
	  
	  grid[k][l][m].bx-=grid[k][l][m].b0x;
	  grid[k][l][m].b0x=mu0*I*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5))
	    +mu0*I*(1.+cos(w*c-Pi))/2.*Pi*Rs*Rs*(3.*xs*ys*sin(alphas)+3.*xs*zs*cos(alphas))/(4.*Pi*pow(rs,5))+IMF_x;
	  grid[k][l][m].bx+=grid[k][l][m].b0x;
	  
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  z=(m-(Grid_Nz+4)/2)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);

	  xs=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  ys=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx+R;
	  zs=(m-(Grid_Nz+4)/2)*dx;
	  
	  rs=sqrt(xs*xs+ys*ys+zs*zs);
	  
	  grid[k][l][m].by-=grid[k][l][m].b0y;
	  grid[k][l][m].b0y=mu0*I*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5))
	    +mu0*I*(1.+cos(w*c-Pi))/2.*Pi*Rs*Rs*(3.*ys*ys*sin(alphas)+3.*ys*zs*cos(alphas)-rs*rs*sin(alphas))/(4.*Pi*pow(rs,5))+IMF_y;
	  grid[k][l][m].by+=grid[k][l][m].b0y;

	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);

	  xs=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  ys=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx+R;
	  zs=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  rs=sqrt(xs*xs+ys*ys+zs*zs);

	  grid[k][l][m].bz-=grid[k][l][m].b0z;
	  grid[k][l][m].b0z=mu0*I*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5))
	    +mu0*I*(1.+cos(w*c-Pi))/2.*Pi*Rs*Rs*(3.*ys*zs*sin(alphas)+3.*zs*zs*cos(alphas)-rs*rs*cos(alphas))/(4.*Pi*pow(rs,5))+IMF_z;
	  grid[k][l][m].bz+=grid[k][l][m].b0z;
	}
      }
    }
  }else{
#pragma omp parallel for private(l,m,x,y,z,r,xs,ys,zs,rs)
    for(k=0;k<Grid_Nx+4;k++){
      for(l=0;l<Grid_Ny+4;l++){
	for(m=0;m<Grid_Nz+4;m++){
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z=(m-(Grid_Nz+4)/2)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);

	  xs=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  ys=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx+R;
	  zs=(m-(Grid_Nz+4)/2)*dx;
	  
	  rs=sqrt(xs*xs+ys*ys+zs*zs);
	  
	  grid[k][l][m].bx-=grid[k][l][m].b0x;
	  grid[k][l][m].b0x=mu0*I*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5))
	    +mu0*I*Pi*Rs*Rs*(3.*xs*ys*sin(alphas)+3.*xs*zs*cos(alphas))/(4.*Pi*pow(rs,5))+IMF_x;
	  grid[k][l][m].bx+=grid[k][l][m].b0x;
	  
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  z=(m-(Grid_Nz+4)/2)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);

	  xs=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  ys=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx+R;
	  zs=(m-(Grid_Nz+4)/2)*dx;
	  
	  rs=sqrt(xs*xs+ys*ys+zs*zs);
	  
	  grid[k][l][m].by-=grid[k][l][m].b0y;
	  grid[k][l][m].b0y=mu0*I*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5))
	    +mu0*I*Pi*Rs*Rs*(3.*ys*ys*sin(alphas)+3.*ys*zs*cos(alphas)-rs*rs*sin(alphas))/(4.*Pi*pow(rs,5))+IMF_y;
	  grid[k][l][m].by+=grid[k][l][m].b0y;

	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);

	  xs=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  ys=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx+R;
	  zs=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  rs=sqrt(xs*xs+ys*ys+zs*zs);

	  grid[k][l][m].bz-=grid[k][l][m].b0z;
	  grid[k][l][m].b0z=mu0*I*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5))
	    +mu0*I*Pi*Rs*Rs*(3.*ys*zs*sin(alphas)+3.*zs*zs*cos(alphas)-rs*rs*cos(alphas))/(4.*Pi*pow(rs,5))+IMF_z;
	  grid[k][l][m].bz+=grid[k][l][m].b0z;
	}
      }
    }
  }

  return(0);
}

int external_current2(const int myrank_x,const int myrank_y,const int c)
{
  const double w=Pi/50.;
  int k,l,m;
  double x,y;

#pragma omp parallel for private(l,m,x,y)
  for(k=0;k<Grid_Nx+4;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;

	if(x*x+y*y<dx*dx&&m==(Grid_Nz+4)/2){
	  grid[k][l][m].jiz0+=I*sin(w*c);
	}
      }
    }
  }

  return(0);
}

int external_current3(const int myrank_x,const int myrank_y,const int c)
{
  const double w=Pi/50000.;
  int i,k,l,m;
  double x,y,z,r;

  if(c<50000){
#pragma omp parallel for private(l,m,x,y,z,r)
    for(k=0;k<Grid_Nx+4;k++){
      for(l=0;l<Grid_Ny+4;l++){
	for(m=0;m<Grid_Nz+4;m++){
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z=(m-(Grid_Nz+4)/2)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);
	  
	  grid[k][l][m].bx-=grid[k][l][m].b0x;
	  grid[k][l][m].b0x=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5))+IMF_x;
	  grid[k][l][m].bx+=grid[k][l][m].b0x;
	  
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  z=(m-(Grid_Nz+4)/2)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);
	  
	  grid[k][l][m].by-=grid[k][l][m].b0y;
	  grid[k][l][m].b0y=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5))+IMF_y;
	  grid[k][l][m].by+=grid[k][l][m].b0y;

	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);

	  grid[k][l][m].bz-=grid[k][l][m].b0z;
	  grid[k][l][m].b0z=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5))+IMF_z;
	  grid[k][l][m].bz+=grid[k][l][m].b0z;
	}
      }
    }
  }else{
#pragma omp parallel for private(l,m,x,y,z,r)
    for(k=0;k<Grid_Nx+4;k++){
      for(l=0;l<Grid_Ny+4;l++){
	for(m=0;m<Grid_Nz+4;m++){
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z=(m-(Grid_Nz+4)/2)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);
	  
	  grid[k][l][m].b0x=mu0*I*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5))+IMF_x;
	  
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  z=(m-(Grid_Nz+4)/2)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);
	  
	  grid[k][l][m].b0y=mu0*I*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5))+IMF_y;
	  
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);
	  
	  grid[k][l][m].b0z=mu0*I*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5))+IMF_z;
	}
      }
    }
  }

  return(0);
}

int external_current4(const int myrank_x,const int myrank_y,const int c)
{
  const double w=Pi/50000.;
  int i,k,l,m;
  double x,y,z,r;

  if(c<50000){
#pragma omp parallel for private(l,m,x,y,z,r)
    for(k=0;k<Grid_Nx+4;k++){
      for(l=0;l<Grid_Ny+4;l++){
	for(m=0;m<Grid_Nz+4;m++){
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z=(m-(Grid_Nz+4)/2)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);
	  
	  grid[k][l][m].bx-=grid[k][l][m].b0x;
	  grid[k][l][m].b0x=(1.+cos(w*c-Pi))/2.*integrate_bx(x,y,z)+IMF_x;
	  grid[k][l][m].bx+=grid[k][l][m].b0x;
	  
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  z=(m-(Grid_Nz+4)/2)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);
	  
	  grid[k][l][m].by-=grid[k][l][m].b0y;
	  grid[k][l][m].b0y=(1.+cos(w*c-Pi))/2.*integrate_by(x,y,z)+IMF_y;
	  grid[k][l][m].by+=grid[k][l][m].b0y;

	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);

	  grid[k][l][m].bz-=grid[k][l][m].b0z;
	  grid[k][l][m].b0z=(1.+cos(w*c-Pi))/2.*integrate_bz(x,y,z)+IMF_z;
	  grid[k][l][m].bz+=grid[k][l][m].b0z;
	}
      }
    }
  }

  return(0);
}

int external_current5(const int myrank_x,const int myrank_y,const int c)
{
  const double w=Pi/50000.;
  int i,k,l,m;
  double x1,y1,z1,r1;
  double x2,y2,z2,r2;
  double x3,y3,z3,r3;
  double x4,y4,z4,r4;
  const double LL=30.;

  if(c<50000){
#pragma omp parallel for private(l,m,x1,y1,z1,r1,x2,y2,z2,r2,x3,y3,z3,r3,x4,y4,z4,r4)
    for(k=0;k<Grid_Nx+4;k++){
      for(l=0;l<Grid_Ny+4;l++){
	for(m=0;m<Grid_Nz+4;m++){
	  x1=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx-LL;
	  y1=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx-LL;
	  z1=(m-(Grid_Nz+4)/2)*dx;
	  
	  r1=sqrt(x1*x1+y1*y1+z1*z1);

	  x2=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx+LL;
	  y2=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx-LL;
	  z2=(m-(Grid_Nz+4)/2)*dx;
	  
	  r2=sqrt(x2*x2+y2*y2+z2*z2);

	  x3=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx-LL;
	  y3=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx+LL;
	  z3=(m-(Grid_Nz+4)/2)*dx;
	  
	  r3=sqrt(x3*x3+y3*y3+z3*z3);

	  x4=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx+LL;
	  y4=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx+LL;
	  z4=(m-(Grid_Nz+4)/2)*dx;
	  
	  r4=sqrt(x4*x4+y4*y4+z4*z4);
	  
	  grid[k][l][m].bx-=grid[k][l][m].b0x;
	  grid[k][l][m].b0x=(mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*x1*y1*sin(alpha)+3.*x1*z1*cos(alpha))/(4.*Pi*pow(r1,5))+
			     mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*x2*y2*sin(alpha)+3.*x2*z2*cos(alpha))/(4.*Pi*pow(r2,5))+
			     mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*x3*y3*sin(alpha)+3.*x3*z3*cos(alpha))/(4.*Pi*pow(r3,5))+
			     mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*x4*y4*sin(alpha)+3.*x4*z4*cos(alpha))/(4.*Pi*pow(r4,5))+
			     IMF_x);
	  grid[k][l][m].bx+=grid[k][l][m].b0x;

	  x1=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx-LL;
	  y1=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx-LL;
	  z1=(m-(Grid_Nz+4)/2)*dx;
	  
	  r1=sqrt(x1*x1+y1*y1+z1*z1);

	  x2=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx+LL;
	  y2=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx-LL;
	  z2=(m-(Grid_Nz+4)/2)*dx;
	  
	  r2=sqrt(x2*x2+y2*y2+z2*z2);

	  x3=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx-LL;
	  y3=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx+LL;
	  z3=(m-(Grid_Nz+4)/2)*dx;
	  
	  r3=sqrt(x3*x3+y3*y3+z3*z3);

	  x4=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx+LL;
	  y4=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx+LL;
	  z4=(m-(Grid_Nz+4)/2)*dx;
	  
	  r4=sqrt(x4*x4+y4*y4+z4*z4);
	  
	  grid[k][l][m].by-=grid[k][l][m].b0y;
	  grid[k][l][m].b0y=(mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y1*y1*sin(alpha)+3.*y1*z1*cos(alpha)-r1*r1*sin(alpha))/(4.*Pi*pow(r1,5))+
			     mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y2*y2*sin(alpha)+3.*y2*z2*cos(alpha)-r2*r2*sin(alpha))/(4.*Pi*pow(r2,5))+
			     mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y3*y3*sin(alpha)+3.*y3*z3*cos(alpha)-r3*r3*sin(alpha))/(4.*Pi*pow(r3,5))+
			     mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y4*y4*sin(alpha)+3.*y4*z4*cos(alpha)-r4*r4*sin(alpha))/(4.*Pi*pow(r4,5))+
			     IMF_y);
	  grid[k][l][m].by+=grid[k][l][m].b0y;

	  x1=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx-LL;
	  y1=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx-LL;
	  z1=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r1=sqrt(x1*x1+y1*y1+z1*z1);

	  x2=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx+LL;
	  y2=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx-LL;
	  z2=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r2=sqrt(x2*x2+y2*y2+z2*z2);

	  x3=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx-LL;
	  y3=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx+LL;
	  z3=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r3=sqrt(x3*x3+y3*y3+z3*z3);

	  x4=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx+LL;
	  y4=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx+LL;
	  z4=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r4=sqrt(x4*x4+y4*y4+z4*z4);

	  grid[k][l][m].bz-=grid[k][l][m].b0z;
	  grid[k][l][m].b0z=(mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y1*z1*sin(alpha)+3.*z1*z1*cos(alpha)-r1*r1*cos(alpha))/(4.*Pi*pow(r1,5))+
			     mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y2*z2*sin(alpha)+3.*z2*z2*cos(alpha)-r2*r2*cos(alpha))/(4.*Pi*pow(r2,5))+
			     mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y3*z3*sin(alpha)+3.*z3*z3*cos(alpha)-r3*r3*cos(alpha))/(4.*Pi*pow(r3,5))+
			     mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y4*z4*sin(alpha)+3.*z4*z4*cos(alpha)-r4*r4*cos(alpha))/(4.*Pi*pow(r4,5))+
			     IMF_z);
	  grid[k][l][m].bz+=grid[k][l][m].b0z;
	}
      }
    }
  }else{
#pragma omp parallel for private(l,m,x1,y1,z1,r1,x2,y2,z2,r2,x3,y3,z3,r3,x4,y4,z4,r4)
    for(k=0;k<Grid_Nx+4;k++){
      for(l=0;l<Grid_Ny+4;l++){
	for(m=0;m<Grid_Nz+4;m++){
	  x1=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx-LL;
	  y1=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx-LL;
	  z1=(m-(Grid_Nz+4)/2)*dx;
	  
	  r1=sqrt(x1*x1+y1*y1+z1*z1);

	  x2=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx+LL;
	  y2=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx-LL;
	  z2=(m-(Grid_Nz+4)/2)*dx;
	  
	  r2=sqrt(x2*x2+y2*y2+z2*z2);

	  x3=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx-LL;
	  y3=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx+LL;
	  z3=(m-(Grid_Nz+4)/2)*dx;
	  
	  r3=sqrt(x3*x3+y3*y3+z3*z3);

	  x4=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx+LL;
	  y4=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx+LL;
	  z4=(m-(Grid_Nz+4)/2)*dx;
	  
	  r4=sqrt(x4*x4+y4*y4+z4*z4);
	  
	  grid[k][l][m].bx-=grid[k][l][m].b0x;
	  grid[k][l][m].b0x=(mu0*I*Pi*R*R*(3.*x1*y1*sin(alpha)+3.*x1*z1*cos(alpha))/(4.*Pi*pow(r1,5))+
			     mu0*I*Pi*R*R*(3.*x2*y2*sin(alpha)+3.*x2*z2*cos(alpha))/(4.*Pi*pow(r2,5))+
			     mu0*I*Pi*R*R*(3.*x3*y3*sin(alpha)+3.*x3*z3*cos(alpha))/(4.*Pi*pow(r3,5))+
			     mu0*I*Pi*R*R*(3.*x4*y4*sin(alpha)+3.*x4*z4*cos(alpha))/(4.*Pi*pow(r4,5))+
			     IMF_x);
	  grid[k][l][m].bx+=grid[k][l][m].b0x;

	  x1=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx-LL;
	  y1=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx-LL;
	  z1=(m-(Grid_Nz+4)/2)*dx;
	  
	  r1=sqrt(x1*x1+y1*y1+z1*z1);

	  x2=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx+LL;
	  y2=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx-LL;
	  z2=(m-(Grid_Nz+4)/2)*dx;
	  
	  r2=sqrt(x2*x2+y2*y2+z2*z2);

	  x3=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx-LL;
	  y3=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx+LL;
	  z3=(m-(Grid_Nz+4)/2)*dx;
	  
	  r3=sqrt(x3*x3+y3*y3+z3*z3);

	  x4=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx+LL;
	  y4=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx+LL;
	  z4=(m-(Grid_Nz+4)/2)*dx;
	  
	  r4=sqrt(x4*x4+y4*y4+z4*z4);
	  
	  grid[k][l][m].by-=grid[k][l][m].b0y;
	  grid[k][l][m].b0y=(mu0*I*Pi*R*R*(3.*y1*y1*sin(alpha)+3.*y1*z1*cos(alpha)-r1*r1*sin(alpha))/(4.*Pi*pow(r1,5))+
			     mu0*I*Pi*R*R*(3.*y2*y2*sin(alpha)+3.*y2*z2*cos(alpha)-r2*r2*sin(alpha))/(4.*Pi*pow(r2,5))+
			     mu0*I*Pi*R*R*(3.*y3*y3*sin(alpha)+3.*y3*z3*cos(alpha)-r3*r3*sin(alpha))/(4.*Pi*pow(r3,5))+
			     mu0*I*Pi*R*R*(3.*y4*y4*sin(alpha)+3.*y4*z4*cos(alpha)-r4*r4*sin(alpha))/(4.*Pi*pow(r4,5))+
			     IMF_y);
	  grid[k][l][m].by+=grid[k][l][m].b0y;

	  x1=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx-LL;
	  y1=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx-LL;
	  z1=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r1=sqrt(x1*x1+y1*y1+z1*z1);

	  x2=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx+LL;
	  y2=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx-LL;
	  z2=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r2=sqrt(x2*x2+y2*y2+z2*z2);

	  x3=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx-LL;
	  y3=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx+LL;
	  z3=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r3=sqrt(x3*x3+y3*y3+z3*z3);

	  x4=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx+LL;
	  y4=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx+LL;
	  z4=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r4=sqrt(x4*x4+y4*y4+z4*z4);

	  grid[k][l][m].bz-=grid[k][l][m].b0z;
	  grid[k][l][m].b0z=(mu0*I*Pi*R*R*(3.*y1*z1*sin(alpha)+3.*z1*z1*cos(alpha)-r1*r1*cos(alpha))/(4.*Pi*pow(r1,5))+
			     mu0*I*Pi*R*R*(3.*y2*z2*sin(alpha)+3.*z2*z2*cos(alpha)-r2*r2*cos(alpha))/(4.*Pi*pow(r2,5))+
			     mu0*I*Pi*R*R*(3.*y3*z3*sin(alpha)+3.*z3*z3*cos(alpha)-r3*r3*cos(alpha))/(4.*Pi*pow(r3,5))+
			     mu0*I*Pi*R*R*(3.*y4*z4*sin(alpha)+3.*z4*z4*cos(alpha)-r4*r4*cos(alpha))/(4.*Pi*pow(r4,5))+
			     IMF_z);
	  grid[k][l][m].bz+=grid[k][l][m].b0z;
	}
      }
    }
  }

  return(0);
}

int external_current6(const int myrank_x,const int myrank_y,const int c)
{
  const double w=Pi/50000.;
  const double wr=Pi/5000.;
  int i,k,l,m;
  double x1,y1,z1,r1,t1,p1;
  double x2,y2,z2,r2,t2,p2;
  double x3,y3,z3,r3,t3,p3;
  double x4,y4,z4,r4,t4,p4;
  const double LL=0.;//2.*dx;//0.;

  t1=t2=t3=t4=wr*c;//Pi/2.;
  /*p1=0.;
  p2=Pi/2.;
  p3=Pi;
  p4=3.*Pi/2.;*/
  p1=0.;
  p2=0.;
  p3=0.;
  p4=0.;

  if(c<50000){
#pragma omp parallel for private(l,m,x1,y1,z1,r1,t1,p1,x2,y2,z2,r2,t2,p2,x3,y3,z3,r3,t3,p3,x4,y4,z4,r4,t4,p4)
    for(k=0;k<Grid_Nx+4;k++){
      for(l=0;l<Grid_Ny+4;l++){
	for(m=0;m<Grid_Nz+4;m++){
	  x1=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx-LL;
	  y1=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z1=(m-(Grid_Nz+4)/2)*dx;
	  
	  r1=sqrt(x1*x1+y1*y1+z1*z1);

	  x2=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  y2=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx-LL;
	  z2=(m-(Grid_Nz+4)/2)*dx;
	  
	  r2=sqrt(x2*x2+y2*y2+z2*z2);

	  x3=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx+LL;
	  y3=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z3=(m-(Grid_Nz+4)/2)*dx;
	  
	  r3=sqrt(x3*x3+y3*y3+z3*z3);

	  x4=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  y4=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx+LL;
	  z4=(m-(Grid_Nz+4)/2)*dx;
	  
	  r4=sqrt(x4*x4+y4*y4+z4*z4);
	  
	  grid[k][l][m].bx-=grid[k][l][m].b0x;
	  grid[k][l][m].b0x=(mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(-sin(t1)*cos(p1)*(-2.*x1*x1+y1*y1+z1*z1)+3.*sin(t1)*sin(p1)*x1*y1+3.*cos(t1)*x1*z1)/(4.*Pi*pow(r1,5))/4.+
			     mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(-sin(t2)*cos(p2)*(-2.*x2*x2+y2*y2+z2*z2)+3.*sin(t2)*sin(p2)*x2*y2+3.*cos(t2)*x2*z2)/(4.*Pi*pow(r2,5))/4.+
			     mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(-sin(t3)*cos(p3)*(-2.*x3*x3+y3*y3+z3*z3)+3.*sin(t3)*sin(p3)*x3*y3+3.*cos(t3)*x3*z3)/(4.*Pi*pow(r3,5))/4.+
			     mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(-sin(t4)*cos(p4)*(-2.*x4*x4+y4*y4+z4*z4)+3.*sin(t4)*sin(p4)*x4*y4+3.*cos(t4)*x4*z4)/(4.*Pi*pow(r4,5))/4.+
			     IMF_x);
	  grid[k][l][m].bx+=grid[k][l][m].b0x;

	  x1=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx-LL;
	  y1=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  z1=(m-(Grid_Nz+4)/2)*dx;
	  
	  r1=sqrt(x1*x1+y1*y1+z1*z1);

	  x2=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y2=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx-LL;
	  z2=(m-(Grid_Nz+4)/2)*dx;
	  
	  r2=sqrt(x2*x2+y2*y2+z2*z2);

	  x3=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx+LL;
	  y3=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  z3=(m-(Grid_Nz+4)/2)*dx;
	  
	  r3=sqrt(x3*x3+y3*y3+z3*z3);

	  x4=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y4=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx+LL;
	  z4=(m-(Grid_Nz+4)/2)*dx;
	  
	  r4=sqrt(x4*x4+y4*y4+z4*z4);
	  
	  grid[k][l][m].by-=grid[k][l][m].b0y;
	  grid[k][l][m].b0y=(mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*sin(t1)*cos(p1)*x1*y1-sin(t1)*sin(p1)*(x1*x1-2.*y1*y1+z1*z1)+3.*cos(t1)*y1*z1)/(4.*Pi*pow(r1,5))/4.+
			     mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*sin(t2)*cos(p2)*x2*y2-sin(t2)*sin(p2)*(x2*x2-2.*y2*y2+z2*z2)+3.*cos(t2)*y2*z2)/(4.*Pi*pow(r2,5))/4.+
			     mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*sin(t3)*cos(p3)*x3*y3-sin(t3)*sin(p3)*(x3*x3-2.*y3*y3+z3*z3)+3.*cos(t3)*y3*z3)/(4.*Pi*pow(r3,5))/4.+
			     mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*sin(t4)*cos(p4)*x4*y4-sin(t4)*sin(p4)*(x4*x4-2.*y4*y4+z4*z4)+3.*cos(t4)*y4*z4)/(4.*Pi*pow(r4,5))/4.+
			     IMF_y);
	  grid[k][l][m].by+=grid[k][l][m].b0y;

	  x1=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx-LL;
	  y1=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z1=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r1=sqrt(x1*x1+y1*y1+z1*z1);

	  x2=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y2=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx-LL;
	  z2=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r2=sqrt(x2*x2+y2*y2+z2*z2);

	  x3=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx+LL;
	  y3=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z3=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r3=sqrt(x3*x3+y3*y3+z3*z3);

	  x4=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y4=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx+LL;
	  z4=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r4=sqrt(x4*x4+y4*y4+z4*z4);

	  grid[k][l][m].bz-=grid[k][l][m].b0z;
	  grid[k][l][m].b0z=(mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*sin(t1)*cos(p1)*x1*z1+3.*sin(t1)*sin(p1)*y1*z1-cos(t1)*(x1*x1+y1*y1-2.*z1*z1))/(4.*Pi*pow(r1,5))/4.+
			     mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*sin(t2)*cos(p2)*x2*z2+3.*sin(t2)*sin(p2)*y2*z2-cos(t2)*(x2*x2+y2*y2-2.*z2*z2))/(4.*Pi*pow(r2,5))/4.+
			     mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*sin(t3)*cos(p3)*x3*z3+3.*sin(t3)*sin(p3)*y3*z3-cos(t3)*(x3*x3+y3*y3-2.*z3*z3))/(4.*Pi*pow(r3,5))/4.+
			     mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*sin(t4)*cos(p4)*x4*z4+3.*sin(t4)*sin(p4)*y4*z4-cos(t4)*(x4*x4+y4*y4-2.*z4*z4))/(4.*Pi*pow(r4,5))/4.+
			     IMF_z);
	  grid[k][l][m].bz+=grid[k][l][m].b0z;
	}
      }
    }
  }else{
#pragma omp parallel for private(l,m,x1,y1,z1,r1,t1,p1,x2,y2,z2,r2,t2,p2,x3,y3,z3,r3,t3,p3,x4,y4,z4,r4,t4,p4)
    for(k=0;k<Grid_Nx+4;k++){
      for(l=0;l<Grid_Ny+4;l++){
	for(m=0;m<Grid_Nz+4;m++){
	  x1=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx-LL;
	  y1=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z1=(m-(Grid_Nz+4)/2)*dx;
	  
	  r1=sqrt(x1*x1+y1*y1+z1*z1);

	  x2=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  y2=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx-LL;
	  z2=(m-(Grid_Nz+4)/2)*dx;
	  
	  r2=sqrt(x2*x2+y2*y2+z2*z2);

	  x3=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx+LL;
	  y3=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z3=(m-(Grid_Nz+4)/2)*dx;
	  
	  r3=sqrt(x3*x3+y3*y3+z3*z3);

	  x4=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  y4=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx+LL;
	  z4=(m-(Grid_Nz+4)/2)*dx;
	  
	  r4=sqrt(x4*x4+y4*y4+z4*z4);
	  
	  grid[k][l][m].bx-=grid[k][l][m].b0x;
	  grid[k][l][m].b0x=(mu0*I*Pi*R*R*(-sin(t1)*cos(p1)*(-2.*x1*x1+y1*y1+z1*z1)+3.*sin(t1)*sin(p1)*x1*y1+3.*cos(t1)*x1*z1)/(4.*Pi*pow(r1,5))/4.+
			     mu0*I*Pi*R*R*(-sin(t2)*cos(p2)*(-2.*x2*x2+y2*y2+z2*z2)+3.*sin(t2)*sin(p2)*x2*y2+3.*cos(t2)*x2*z2)/(4.*Pi*pow(r2,5))/4.+
			     mu0*I*Pi*R*R*(-sin(t3)*cos(p3)*(-2.*x3*x3+y3*y3+z3*z3)+3.*sin(t3)*sin(p3)*x3*y3+3.*cos(t3)*x3*z3)/(4.*Pi*pow(r3,5))/4.+
			     mu0*I*Pi*R*R*(-sin(t4)*cos(p4)*(-2.*x4*x4+y4*y4+z4*z4)+3.*sin(t4)*sin(p4)*x4*y4+3.*cos(t4)*x4*z4)/(4.*Pi*pow(r4,5))/4.+
			     IMF_x);
	  grid[k][l][m].bx+=grid[k][l][m].b0x;

	  x1=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx-LL;
	  y1=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  z1=(m-(Grid_Nz+4)/2)*dx;
	  
	  r1=sqrt(x1*x1+y1*y1+z1*z1);

	  x2=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y2=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx-LL;
	  z2=(m-(Grid_Nz+4)/2)*dx;
	  
	  r2=sqrt(x2*x2+y2*y2+z2*z2);

	  x3=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx+LL;
	  y3=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  z3=(m-(Grid_Nz+4)/2)*dx;
	  
	  r3=sqrt(x3*x3+y3*y3+z3*z3);

	  x4=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y4=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx+LL;
	  z4=(m-(Grid_Nz+4)/2)*dx;
	  
	  r4=sqrt(x4*x4+y4*y4+z4*z4);
	  
	  grid[k][l][m].by-=grid[k][l][m].b0y;
	  grid[k][l][m].b0y=(mu0*I*Pi*R*R*(3.*sin(t1)*cos(p1)*x1*y1-sin(t1)*sin(p1)*(x1*x1-2.*y1*y1+z1*z1)+3.*cos(t1)*y1*z1)/(4.*Pi*pow(r1,5))/4.+
			     mu0*I*Pi*R*R*(3.*sin(t2)*cos(p2)*x2*y2-sin(t2)*sin(p2)*(x2*x2-2.*y2*y2+z2*z2)+3.*cos(t2)*y2*z2)/(4.*Pi*pow(r2,5))/4.+
			     mu0*I*Pi*R*R*(3.*sin(t3)*cos(p3)*x3*y3-sin(t3)*sin(p3)*(x3*x3-2.*y3*y3+z3*z3)+3.*cos(t3)*y3*z3)/(4.*Pi*pow(r3,5))/4.+
			     mu0*I*Pi*R*R*(3.*sin(t4)*cos(p4)*x4*y4-sin(t4)*sin(p4)*(x4*x4-2.*y4*y4+z4*z4)+3.*cos(t4)*y4*z4)/(4.*Pi*pow(r4,5))/4.+
			     IMF_y);
	  grid[k][l][m].by+=grid[k][l][m].b0y;

	  x1=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx-LL;
	  y1=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z1=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r1=sqrt(x1*x1+y1*y1+z1*z1);

	  x2=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y2=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx-LL;
	  z2=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r2=sqrt(x2*x2+y2*y2+z2*z2);

	  x3=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx+LL;
	  y3=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z3=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r3=sqrt(x3*x3+y3*y3+z3*z3);

	  x4=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y4=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx+LL;
	  z4=(m-(Grid_Nz+4)/2+0.5)*dx;
	  
	  r4=sqrt(x4*x4+y4*y4+z4*z4);

	  grid[k][l][m].bz-=grid[k][l][m].b0z;
	  grid[k][l][m].b0z=(mu0*I*Pi*R*R*(3.*sin(t1)*cos(p1)*x1*z1+3.*sin(t1)*sin(p1)*y1*z1-cos(t1)*(x1*x1+y1*y1-2.*z1*z1))/(4.*Pi*pow(r1,5))/4.+
			     mu0*I*Pi*R*R*(3.*sin(t2)*cos(p2)*x2*z2+3.*sin(t2)*sin(p2)*y2*z2-cos(t2)*(x2*x2+y2*y2-2.*z2*z2))/(4.*Pi*pow(r2,5))/4.+
			     mu0*I*Pi*R*R*(3.*sin(t3)*cos(p3)*x3*z3+3.*sin(t3)*sin(p3)*y3*z3-cos(t3)*(x3*x3+y3*y3-2.*z3*z3))/(4.*Pi*pow(r3,5))/4.+
			     mu0*I*Pi*R*R*(3.*sin(t4)*cos(p4)*x4*z4+3.*sin(t4)*sin(p4)*y4*z4-cos(t4)*(x4*x4+y4*y4-2.*z4*z4))/(4.*Pi*pow(r4,5))/4.+
			     IMF_z);
	  grid[k][l][m].bz+=grid[k][l][m].b0z;
	}
      }
    }
  }

  return(0);
}

int external_current7(const int myrank_x,const int myrank_y,const int c)
{
  int i,k,l,m;
  double x,y,z,r;

#pragma omp parallel for private(l,m,x,y,z,r)
  for(k=0;k<Grid_Nx+4;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	z=(m-(Grid_Nz+4)/2)*dx;

	r=sqrt(x*x+y*y+z*z);

	if(r<210.*dx){
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z=(m-(Grid_Nz+4)/2)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);
	  
	  grid[k][l][m].b0x=mu0*I*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5))+IMF_x;
	  grid[k][l][m].bx=grid[k][l][m].b0x;
	  
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  z=(m-(Grid_Nz+4)/2)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);
	  
	  grid[k][l][m].b0y=mu0*I*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5))+IMF_y;
	  grid[k][l][m].by=grid[k][l][m].b0y;
	
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z=(m-(Grid_Nz+4)/2+0.5)*dx;
	
	  r=sqrt(x*x+y*y+z*z);
	
	  grid[k][l][m].b0z=mu0*I*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5))+IMF_z;
	  grid[k][l][m].bz=grid[k][l][m].b0z;
	}else{
	  grid[k][l][m].bx=IMF_x;
	  grid[k][l][m].by=IMF_y;
	  grid[k][l][m].bz=IMF_z;
	  grid[k][l][m].b0x=IMF_x;
	  grid[k][l][m].b0y=IMF_y;
	  grid[k][l][m].b0z=IMF_z;
	}
      }
    }
  }

  return(0);
}

int external_current9(const int myrank_x,const int myrank_y,const int c)
{
  int i,k,l,m;
  double x,y,z,r;

#pragma omp parallel for private(l,m,x,y,z,r)
  for(k=0;k<Grid_Nx+4;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	z=(m-(Grid_Nz+4)/2)*dx;

	r=sqrt(x*x+y*y+z*z);

	if(r<210.*dx){
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2+0.5)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z=(m-(Grid_Nz+4)/2)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);
	  
	  grid[k][l][m].b0x=mu0*I*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5))+IMF_x;
	  grid[k][l][m].bx=grid[k][l][m].b0x;
	  
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2+0.5)*dx;
	  z=(m-(Grid_Nz+4)/2)*dx;
	  
	  r=sqrt(x*x+y*y+z*z);
	  
	  grid[k][l][m].b0y=mu0*I*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5))+IMF_y;
	  grid[k][l][m].by=grid[k][l][m].b0y;
	
	  x=(k+myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
	  y=(l+myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
	  z=(m-(Grid_Nz+4)/2+0.5)*dx;
	
	  r=sqrt(x*x+y*y+z*z);
	
	  grid[k][l][m].b0z=mu0*I*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5))+IMF_z;
	  grid[k][l][m].bz=grid[k][l][m].b0z;
	}else{
	  grid[k][l][m].bx=IMF_x;
	  grid[k][l][m].by=IMF_y;
	  grid[k][l][m].bz=IMF_z;
	  grid[k][l][m].b0x=IMF_x;
	  grid[k][l][m].b0y=IMF_y;
	  grid[k][l][m].b0z=IMF_z;
	}
      }
    }
  }

  return(0);
}

/*******************************************************************
magnetic field by coil
 bx
******************************************************************/
double integrate_bx(double k,double l,double m)
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

  gsl_integration_qags(&F,0.,2*Pi,0.0001,0.0001,10000,ws,&result,&error);

  gsl_integration_workspace_free(ws);
  return(mu0*I/4./Pi*result);
}

/*******************************************************************
magnetic field by coil
 by
******************************************************************/
double integrate_by(double k,double l,double m)
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

  gsl_integration_qags(&F,0.,2*Pi,0.0001,0.0001,10000,ws,&result,&error);

  gsl_integration_workspace_free(ws);
  return(mu0*I/4./Pi*result);
}

/*******************************************************************
magnetic field by coil
 bz
******************************************************************/
double integrate_bz(double k,double l,double m)
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

  gsl_integration_qags(&F,0.,2*Pi,0.0001,0.0001,10000,ws,&result,&error);

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

  x=(double)(p->k);
  y=(double)(p->l);
  z=(double)(p->m);

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

  x=(double)(p->k);
  y=(double)(p->l);
  z=(double)(p->m);

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

  x=(double)(p->k);
  y=(double)(p->l);
  z=(double)(p->m);

  r=R*(R*cos(alpha)-y*sin(w)-x*cos(w)*cos(alpha))
    *pow(gsl_pow_2(R)+gsl_pow_2(x)+gsl_pow_2(y)+gsl_pow_2(z)
	 -2*R*(x*cos(w)+y*sin(w)*cos(alpha)+z*sin(w)*sin(alpha)),-1.5);

  return(r);
}

int output(const int myrank_x,const int myrank_y,const int c)
{
  int k,l,m;
  FILE *fp1;
  char filename[256];
  
  sprintf(filename,"fdtd%d-%d-%d-%d.dat",version,myrank_x,myrank_y,c);
  fp1=fopen(filename,"w");
  
  for(k=2;k<Grid_Nx+2;k++){
    for(l=2;l<Grid_Ny+2;l++){
      for(m=2;m<Grid_Nz+2;m++){
	fprintf(fp1,"%.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E\n",
		(k+myrank_x*Grid_Nx-2-PROCESS_Nx*Grid_Nx/2)*dx,
		(l+myrank_y*Grid_Ny-2-PROCESS_Ny*Grid_Ny/2)*dx,
		(m-(Grid_Nz+4)/2)*dx,
		grid[k][l][m].ni,
		grid[k][l][m].ne,
		grid[k][l][m].ni-grid[k][l][m].ne,
		(grid[k+1][l][m].ex+grid[k][l][m].ex)/2.,
		(grid[k][l+1][m].ey+grid[k][l][m].ey)/2.,
		(grid[k][l][m+1].ez+grid[k][l][m].ez)/2.,
		(grid[k][l][m].bx+grid[k][l+1][m].bx+grid[k][l][m+1].bx+grid[k][l+1][m+1].bx)/4.,
		(grid[k][l][m].by+grid[k+1][l][m].by+grid[k][l][m+1].by+grid[k+1][l][m+1].by)/4.,
		(grid[k][l][m].bz+grid[k+1][l][m].bz+grid[k][l+1][m].bz+grid[k+1][l+1][m].bz)/4.,
		grid[k][l][m].jix0,
		grid[k][l][m].jiy0,
		grid[k][l][m].jiz0,
		grid[k][l][m].jex0,
		grid[k][l][m].jey0,
		grid[k][l][m].jez0,
		/*sqrt(pow(grid[k][l][m].jix0-grid[k][l][m].jex0,2)
		     +pow(grid[k][l][m].jiy0-grid[k][l][m].jey0,2)
		     +pow(grid[k][l][m].jiz0-grid[k][l][m].jez0,2))*/
		grid[k][l][m].phi,/////
		sqrt(pow((grid[k][l][m].bx+grid[k][l+1][m].bx+grid[k][l][m+1].bx+grid[k][l+1][m+1].bx)/4.,2)
		     +pow((grid[k][l][m].by+grid[k+1][l][m].by+grid[k][l][m+1].by+grid[k+1][l][m+1].by)/4.,2)
		     +pow((grid[k][l][m].bz+grid[k+1][l][m].bz+grid[k][l+1][m].bz+grid[k+1][l+1][m].bz)/4.,2)),
		(grid[k][l][m].bx+grid[k][l+1][m].bx+grid[k][l][m+1].bx+grid[k][l+1][m+1].bx)/4.-(grid[k][l][m].b0x+grid[k][l+1][m].b0x+grid[k][l][m+1].b0x+grid[k][l+1][m+1].b0x)/4.,
		(grid[k][l][m].by+grid[k+1][l][m].by+grid[k][l][m+1].by+grid[k+1][l][m+1].by)/4.-(grid[k][l][m].b0y+grid[k+1][l][m].b0y+grid[k][l][m+1].b0y+grid[k+1][l][m+1].b0y)/4.,
		(grid[k][l][m].bz+grid[k+1][l][m].bz+grid[k][l+1][m].bz+grid[k+1][l+1][m].bz)/4.-(grid[k][l][m].b0z+grid[k+1][l][m].b0z+grid[k][l+1][m].b0z+grid[k+1][l+1][m].b0z)/4.);
	
      }
    }
  }
  
  fclose(fp1);

  return(0);
}

int output_tecplot(const int myrank_x,const int myrank_y,const int c)
{
  int k,l,m;
  FILE *fp1;
  char filename[256];
  
  sprintf(filename,"fdtd%d-%d-%d-%d.dat",version,myrank_x,myrank_y,c);
  fp1=fopen(filename,"w");
  
  fprintf(fp1,"VARIABLES = \"X[m]\" \"Y[m]\" \"Z[m]\" \"ni[/m3]\" \"ne[/m3]\" \"dens[/m3]\" \"Ex[V/m]\" \"Ey[V/m]\" \"Ez[V/m]\" \"Bx[T]\" \"By[T]\" \"Bz[T]\" \"jix[A/m2]\" \"jiy[A/m2]\" \"jiz[A/m2]\" \"jex[A/m2]\" \"jey[A/m2]\" \"jez[A/m2]\" \"j[A/m2]\" \"phi[V]\" \"B[T]\" \"Bpx[T]\" \"Bpy[T]\" \"Bpz[T]\"\n");
  fprintf(fp1,"ZONE T=\"STP:%d\", STRANDID=1, SOLUTIONTIME=%d, I=%d, J=%d, K=%d\n",c,c,Grid_Nz,Grid_Ny+1,Grid_Nx+1);
  
  for(k=2;k<Grid_Nx+3;k++){
    for(l=2;l<Grid_Ny+3;l++){
      for(m=2;m<Grid_Nz+2;m++){
	fprintf(fp1,"%.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E\n",
		(k+myrank_x*Grid_Nx-2-(PROCESS_Nx*Grid_Nx)/2)*dx,
		(l+myrank_y*Grid_Ny-2-(PROCESS_Ny*Grid_Ny)/2)*dx,
		(m-(Grid_Nz+4)/2)*dx,
		grid[k][l][m].ni,
		grid[k][l][m].ne,
		grid[k][l][m].ni-grid[k][l][m].ne,
		(grid[k+1][l][m].ex+grid[k][l][m].ex)/2.,
		(grid[k][l+1][m].ey+grid[k][l][m].ey)/2.,
		(grid[k][l][m+1].ez+grid[k][l][m].ez)/2.,
		(grid[k][l][m].bx+grid[k][l+1][m].bx+grid[k][l][m+1].bx+grid[k][l+1][m+1].bx)/4.,
		(grid[k][l][m].by+grid[k+1][l][m].by+grid[k][l][m+1].by+grid[k+1][l][m+1].by)/4.,
		(grid[k][l][m].bz+grid[k+1][l][m].bz+grid[k][l+1][m].bz+grid[k+1][l+1][m].bz)/4.,
		(grid[k+1][l][m].jix0+grid[k][l][m].jix0)/2.,
		(grid[k][l+1][m].jiy0+grid[k][l][m].jiy0)/2.,
		(grid[k][l][m+1].jiz0+grid[k][l][m].jiz0)/2.,
		(grid[k+1][l][m].jex0+grid[k][l][m].jex0)/2.,
		(grid[k][l+1][m].jey0+grid[k][l][m].jey0)/2.,
		(grid[k][l][m+1].jez0+grid[k][l][m].jez0)/2.,
		sqrt(pow((grid[k+1][l][m].jix0+grid[k][l][m].jix0)/2.-(grid[k+1][l][m].jex0+grid[k][l][m].jex0)/2.,2)
		     +pow((grid[k][l+1][m].jiy0+grid[k][l][m].jiy0)/2.-(grid[k][l+1][m].jey0+grid[k][l][m].jey0)/2.,2)
		     +pow((grid[k][l][m+1].jiz0+grid[k][l][m].jiz0)/2.-(grid[k][l][m+1].jez0+grid[k][l][m].jez0)/2.,2)),
		grid[k][l][m].phi,
		sqrt(pow((grid[k][l][m].bx+grid[k][l+1][m].bx+grid[k][l][m+1].bx+grid[k][l+1][m+1].bx)/4.,2)
		     +pow((grid[k][l][m].by+grid[k+1][l][m].by+grid[k][l][m+1].by+grid[k+1][l][m+1].by)/4.,2)
		     +pow((grid[k][l][m].bz+grid[k+1][l][m].bz+grid[k][l+1][m].bz+grid[k+1][l+1][m].bz)/4.,2)),
		(grid[k][l][m].bx+grid[k][l+1][m].bx+grid[k][l][m+1].bx+grid[k][l+1][m+1].bx)/4.-(grid[k][l][m].b0x+grid[k][l+1][m].b0x+grid[k][l][m+1].b0x+grid[k][l+1][m+1].b0x)/4.,
		(grid[k][l][m].by+grid[k+1][l][m].by+grid[k][l][m+1].by+grid[k+1][l][m+1].by)/4.-(grid[k][l][m].b0y+grid[k+1][l][m].b0y+grid[k][l][m+1].b0y+grid[k+1][l][m+1].b0y)/4.,
		(grid[k][l][m].bz+grid[k+1][l][m].bz+grid[k][l+1][m].bz+grid[k+1][l+1][m].bz)/4.-(grid[k][l][m].b0z+grid[k+1][l][m].b0z+grid[k][l+1][m].b0z+grid[k+1][l+1][m].b0z)/4.);
	
      }
      fprintf(fp1,"\n");
    }
  }
  
  fclose(fp1);

  return(0);
}

int thrust_f(const int myrank_x,const int myrank_y,const int c,Particle *ion[],Particle *ion_start[],Particle *electron[],Particle *electron_start[])
{
  int i,k,l,m;
  FILE *fp1;
  char filename[256];
  double ex,ey,ez;
  double bx,by,bz;

  double x,y,z,r,w;

  double F[27];  //Fx,Fy,Fz;Fxie,Fyie,Fzie;Fxib,Fyib,Fzib;Fxee,Fyee,Fzee;Fxeb,Fyeb,Fzeb;Fxie2,Fyie2,Fzie2;Fxib2,Fyib2,Fzib2;Fxee2,Fyee2,Fzee2;Fxeb2,Fyeb2,Fzeb2
  double F_r[27];

  double E[6];//E,B,ion,electron,ion2,electron2
  double E_r[6];

  double Vox,Voy,Voz;

  w=Pi/50000.;

  for(i=0;i<27;i++){
    F[i]=0.;
  }

  for(i=0;i<6;i++){
    E[i]=0.;
  }

  for(k=2;k<Grid_Nx+2;k++){
    for(l=2;l<Grid_Ny+2;l++){
      for(m=2;m<Grid_Nz+2;m++){
	if(myrank_x*Grid_Nx+k-2<PROCESS_Nx*Grid_Nx*0.8&&myrank_x*Grid_Nx+k-2>PROCESS_Nx*Grid_Nx*0.2&&
	   myrank_y*Grid_Ny+l-2<PROCESS_Ny*Grid_Ny*0.8&&myrank_y*Grid_Ny+l-2>PROCESS_Ny*Grid_Ny*0.2&&
	   m-2<Grid_Nz*0.8&&m-2>Grid_Nz*0.2){
	  E[0]+=0.5*(grid[k][l][m].ex*grid[k][l][m].ex+grid[k][l][m].ey*grid[k][l][m].ey+grid[k][l][m].ez*grid[k][l][m].ez)*e0;
	  E[1]+=0.5*(grid[k][l][m].bx*grid[k][l][m].bx+grid[k][l][m].by*grid[k][l][m].by+grid[k][l][m].bz*grid[k][l][m].bz)/mu0;
	}
      }
    }
  }

  for(i=0;i<THREAD_N;i++){
    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){
      if(myrank_x*Grid_Nx*dx+ion[i]->x<PROCESS_Nx*Grid_Nx*dx*0.8&&myrank_x*Grid_Nx*dx+ion[i]->x>PROCESS_Nx*Grid_Nx*dx*0.2&&
	 myrank_y*Grid_Ny*dx+ion[i]->y<PROCESS_Ny*Grid_Ny*dx*0.8&&myrank_y*Grid_Ny*dx+ion[i]->y>PROCESS_Ny*Grid_Ny*dx*0.2&&
	 ion[i]->z<Grid_Nz*dx*0.8&&ion[i]->z>Grid_Nz*dx*0.2){
	shape_func_v2(ion[i],&ex,&ey,&ez,&bx,&by,&bz);

	Vox=-(ion[i]->z+(-Grid_Nz/2-2)*dx)*omega_y-(ion[i]->y+(myrank_y*Grid_Ny-PROCESS_Ny*Grid_Ny/2-2)*dx)*omega_z;
	Voy=(ion[i]->x+(myrank_x*Grid_Nx-PROCESS_Nx*Grid_Nx/2-2)*dx)*omega_z;
	Voz=(ion[i]->x+(myrank_x*Grid_Nx-PROCESS_Nx*Grid_Nx/2-2)*dx)*omega_y;

	DIPOLE(
	       if(c<50000){
		 x=ion[i]->x+(myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
		 y=ion[i]->y+(myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
		 z=ion[i]->z+(-(Grid_Nz+4)/2)*dx;
		 
		 r=sqrt(x*x+y*y+z*z);
		 
		 bx+=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5))+IMF_x;
		 by+=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5))+IMF_y;
		 bz+=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5))+IMF_z;
	       }else{
		 x=ion[i]->x+(myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
		 y=ion[i]->y+(myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
		 z=ion[i]->z+(-(Grid_Nz+4)/2)*dx;
		 
		 r=sqrt(x*x+y*y+z*z);
		 
		 bx+=mu0*I*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5))+IMF_x;
		 by+=mu0*I*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5))+IMF_y;
		 bz+=mu0*I*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5))+IMF_z;
	       }
	       );

	F[0]+=q*(ex+(ion[i]->vy+Voy)*bz-(ion[i]->vz+Voz)*by)*ion[i]->n*dx*dx*dx;
	F[1]+=q*(ey+(ion[i]->vz+Voz)*bx-(ion[i]->vx+Vox)*bz)*ion[i]->n*dx*dx*dx;
	F[2]+=q*(ez+(ion[i]->vx+Vox)*by-(ion[i]->vy+Voy)*bx)*ion[i]->n*dx*dx*dx;

	if(ion[i]->flag>0){
	  F[3]+=q*ex*ion[i]->n*dx*dx*dx;
	  F[4]+=q*ey*ion[i]->n*dx*dx*dx;
	  F[5]+=q*ez*ion[i]->n*dx*dx*dx;
	  F[6]+=q*((ion[i]->vy+Voy)*bz-(ion[i]->vz+Voz)*by)*ion[i]->n*dx*dx*dx;
	  F[7]+=q*((ion[i]->vz+Voz)*bx-(ion[i]->vx+Vox)*bz)*ion[i]->n*dx*dx*dx;
	  F[8]+=q*((ion[i]->vx+Vox)*by-(ion[i]->vy+Voy)*bx)*ion[i]->n*dx*dx*dx;

	  E[2]+=0.5*mi*ion[i]->n*dx*dx*dx*(ion[i]->vx*ion[i]->vx+ion[i]->vy*ion[i]->vy+ion[i]->vz*ion[i]->vz);
	}else if(ion[i]->flag<0){
	  F[15]+=q*ex*ion[i]->n*dx*dx*dx;
	  F[16]+=q*ey*ion[i]->n*dx*dx*dx;
	  F[17]+=q*ez*ion[i]->n*dx*dx*dx;
	  F[18]+=q*((ion[i]->vy+Voy)*bz-(ion[i]->vz+Voz)*by)*ion[i]->n*dx*dx*dx;
	  F[19]+=q*((ion[i]->vz+Voz)*bx-(ion[i]->vx+Vox)*bz)*ion[i]->n*dx*dx*dx;
	  F[20]+=q*((ion[i]->vx+Vox)*by-(ion[i]->vy+Voy)*bx)*ion[i]->n*dx*dx*dx;

	  E[4]+=0.5*mi*ion[i]->n*dx*dx*dx*(ion[i]->vx*ion[i]->vx+ion[i]->vy*ion[i]->vy+ion[i]->vz*ion[i]->vz);
	}
      }
      ion[i]=ion[i]->next_particle;
    }

    electron[i]=electron_start[i];
    while(electron[i]->next_particle!=NULL){
      if(myrank_x*Grid_Nx*dx+electron[i]->x<PROCESS_Nx*Grid_Nx*dx*0.8&&myrank_x*Grid_Nx*dx+electron[i]->x>PROCESS_Nx*Grid_Nx*dx*0.2&&
	 myrank_y*Grid_Ny*dx+electron[i]->y<PROCESS_Ny*Grid_Ny*dx*0.8&&myrank_y*Grid_Ny*dx+electron[i]->y>PROCESS_Ny*Grid_Ny*dx*0.2&&
	 electron[i]->z<Grid_Nz*dx*0.8&&electron[i]->z>Grid_Nz*dx*0.2){
	shape_func_v2(electron[i],&ex,&ey,&ez,&bx,&by,&bz);

	Vox=-(electron[i]->z+(-Grid_Nz/2-2)*dx)*omega_y-(electron[i]->y+(myrank_y*Grid_Ny-PROCESS_Ny*Grid_Ny/2-2)*dx)*omega_z;
	Voy=(electron[i]->x+(myrank_x*Grid_Nx-PROCESS_Nx*Grid_Nx/2-2)*dx)*omega_z;
	Voz=(electron[i]->x+(myrank_x*Grid_Nx-PROCESS_Nx*Grid_Nx/2-2)*dx)*omega_y;

	DIPOLE(
	       if(c<50000){
		 x=electron[i]->x+(myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
		 y=electron[i]->y+(myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
		 z=electron[i]->z+(-(Grid_Nz+4)/2)*dx;
		 
		 r=sqrt(x*x+y*y+z*z);
		 
		 bx+=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5))+IMF_x;
		 by+=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5))+IMF_y;
		 bz+=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5))+IMF_z;
	       }else{
		 x=electron[i]->x+(myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
		 y=electron[i]->y+(myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
		 z=electron[i]->z+(-(Grid_Nz+4)/2)*dx;
		 
		 r=sqrt(x*x+y*y+z*z);
		 
		 bx+=mu0*I*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5))+IMF_x;
		 by+=mu0*I*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5))+IMF_y;
		 bz+=mu0*I*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5))+IMF_z;
	       }
	       );

	F[0]-=q*(ex+(electron[i]->vy+Voy)*bz-(electron[i]->vz+Voz)*by)*electron[i]->n*dx*dx*dx;
	F[1]-=q*(ey+(electron[i]->vz+Voz)*bx-(electron[i]->vx+Vox)*bz)*electron[i]->n*dx*dx*dx;
	F[2]-=q*(ez+(electron[i]->vx+Vox)*by-(electron[i]->vy+Voy)*bx)*electron[i]->n*dx*dx*dx;
	
	if(electron[i]->flag>0){
	  F[9]-=q*ex*electron[i]->n*dx*dx*dx;
	  F[10]-=q*ey*electron[i]->n*dx*dx*dx;
	  F[11]-=q*ez*electron[i]->n*dx*dx*dx;
	  F[12]-=q*((electron[i]->vy+Voy)*bz-(electron[i]->vz+Voz)*by)*electron[i]->n*dx*dx*dx;
	  F[13]-=q*((electron[i]->vz+Voz)*bx-(electron[i]->vx+Vox)*bz)*electron[i]->n*dx*dx*dx;
	  F[14]-=q*((electron[i]->vx+Vox)*by-(electron[i]->vy+Voy)*bx)*electron[i]->n*dx*dx*dx;

	  E[3]+=0.5*me*electron[i]->n*dx*dx*dx*(electron[i]->vx*electron[i]->vx+electron[i]->vy*electron[i]->vy+electron[i]->vz*electron[i]->vz);
	}else if(electron[i]->flag<0){
	  F[21]-=q*ex*electron[i]->n*dx*dx*dx;
	  F[22]-=q*ey*electron[i]->n*dx*dx*dx;
	  F[23]-=q*ez*electron[i]->n*dx*dx*dx;
	  F[24]-=q*((electron[i]->vy+Voy)*bz-(electron[i]->vz+Voz)*by)*electron[i]->n*dx*dx*dx;
	  F[25]-=q*((electron[i]->vz+Voz)*bx-(electron[i]->vx+Vox)*bz)*electron[i]->n*dx*dx*dx;
	  F[26]-=q*((electron[i]->vx+Vox)*by-(electron[i]->vy+Voy)*bx)*electron[i]->n*dx*dx*dx;

	  E[5]+=0.5*me*electron[i]->n*dx*dx*dx*(electron[i]->vx*electron[i]->vx+electron[i]->vy*electron[i]->vy+electron[i]->vz*electron[i]->vz);
	}
      }
      electron[i]=electron[i]->next_particle;
    }
  }

  MPI_Allreduce(F,F_r,27,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(E,E_r,6,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  if(myrank_x==0&&myrank_y==0){
    sprintf(filename,"thrust%d.txt",version);
    fp1=fopen(filename,"a");
    
    if(fprintf(fp1,"%d %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.8E %.8E %.8E %.8E %.8E %.8E\n",c,F_r[0],F_r[1],F_r[2],F_r[3],F_r[4],F_r[5],F_r[6],F_r[7],F_r[8],F_r[9],F_r[10],F_r[11],F_r[12],F_r[13],F_r[14],F_r[15],F_r[16],F_r[17],F_r[18],F_r[19],F_r[20],F_r[21],F_r[22],F_r[23],F_r[24],F_r[25],F_r[26],E_r[0],E_r[1],E_r[2],E_r[3],E_r[4],E_r[5])<0){
      exit(-1);
    }

    fclose(fp1);
  }

  return(0);
}

int thrust_fave(const int myrank_x,const int myrank_y,const int c,Particle *ion[],Particle *ion_start[],Particle *electron[],Particle *electron_start[])
{
  int i,k,l,m;
  FILE *fp1;
  char filename[256];
  double ex,ey,ez;
  double bx,by,bz;

  double x,y,z,r,w;

  double F[27];  //Fx,Fy,Fz;Fxie,Fyie,Fzie;Fxib,Fyib,Fzib;Fxee,Fyee,Fzee;Fxeb,Fyeb,Fzeb;Fxie2,Fyie2,Fzie2;Fxib2,Fyib2,Fzib2;Fxee2,Fyee2,Fzee2;Fxeb2,Fyeb2,Fzeb2
  double F_r[27];

  double Vox,Voy,Voz;

  w=Pi/50000.;

  for(i=0;i<27;i++){
    F[i]=0.;
  }

  for(i=0;i<THREAD_N;i++){
    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){
      if(myrank_x*Grid_Nx*dx+ion[i]->x<PROCESS_Nx*Grid_Nx*dx*0.8&&myrank_x*Grid_Nx*dx+ion[i]->x>PROCESS_Nx*Grid_Nx*dx*0.2&&
	 myrank_y*Grid_Ny*dx+ion[i]->y<PROCESS_Ny*Grid_Ny*dx*0.8&&myrank_y*Grid_Ny*dx+ion[i]->y>PROCESS_Ny*Grid_Ny*dx*0.2&&
	 ion[i]->z<Grid_Nz*dx*0.8&&ion[i]->z>Grid_Nz*dx*0.2){
	shape_func_v2(ion[i],&ex,&ey,&ez,&bx,&by,&bz);

	Vox=-(ion[i]->z+(-Grid_Nz/2-2)*dx)*omega_y-(ion[i]->y+(myrank_y*Grid_Ny-PROCESS_Ny*Grid_Ny/2-2)*dx)*omega_z;
	Voy=(ion[i]->x+(myrank_x*Grid_Nx-PROCESS_Nx*Grid_Nx/2-2)*dx)*omega_z;
	Voz=(ion[i]->x+(myrank_x*Grid_Nx-PROCESS_Nx*Grid_Nx/2-2)*dx)*omega_y;

	DIPOLE(
	       if(c<50000){
		 x=ion[i]->x+(myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
		 y=ion[i]->y+(myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
		 z=ion[i]->z+(-(Grid_Nz+4)/2)*dx;
		 
		 r=sqrt(x*x+y*y+z*z);
		 
		 bx+=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5))+IMF_x;
		 by+=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5))+IMF_y;
		 bz+=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5))+IMF_z;
	       }else{
		 x=ion[i]->x+(myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
		 y=ion[i]->y+(myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
		 z=ion[i]->z+(-(Grid_Nz+4)/2)*dx;
		 
		 r=sqrt(x*x+y*y+z*z);
		 
		 bx+=mu0*I*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5))+IMF_x;
		 by+=mu0*I*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5))+IMF_y;
		 bz+=mu0*I*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5))+IMF_z;
	       }
	       );

	F[0]+=q*(ex+(ion[i]->vy+Voy)*bz-(ion[i]->vz+Voz)*by)*ion[i]->n*dx*dx*dx;
	F[1]+=q*(ey+(ion[i]->vz+Voz)*bx-(ion[i]->vx+Vox)*bz)*ion[i]->n*dx*dx*dx;
	F[2]+=q*(ez+(ion[i]->vx+Vox)*by-(ion[i]->vy+Voy)*bx)*ion[i]->n*dx*dx*dx;

	if(ion[i]->flag>0){
	  F[3]+=q*ex*ion[i]->n*dx*dx*dx;
	  F[4]+=q*ey*ion[i]->n*dx*dx*dx;
	  F[5]+=q*ez*ion[i]->n*dx*dx*dx;
	  F[6]+=q*((ion[i]->vy+Voy)*bz-(ion[i]->vz+Voz)*by)*ion[i]->n*dx*dx*dx;
	  F[7]+=q*((ion[i]->vz+Voz)*bx-(ion[i]->vx+Vox)*bz)*ion[i]->n*dx*dx*dx;
	  F[8]+=q*((ion[i]->vx+Vox)*by-(ion[i]->vy+Voy)*bx)*ion[i]->n*dx*dx*dx;
	}else if(ion[i]->flag<0){
	  F[15]+=q*ex*ion[i]->n*dx*dx*dx;
	  F[16]+=q*ey*ion[i]->n*dx*dx*dx;
	  F[17]+=q*ez*ion[i]->n*dx*dx*dx;
	  F[18]+=q*((ion[i]->vy+Voy)*bz-(ion[i]->vz+Voz)*by)*ion[i]->n*dx*dx*dx;
	  F[19]+=q*((ion[i]->vz+Voz)*bx-(ion[i]->vx+Vox)*bz)*ion[i]->n*dx*dx*dx;
	  F[20]+=q*((ion[i]->vx+Vox)*by-(ion[i]->vy+Voy)*bx)*ion[i]->n*dx*dx*dx;
	}
      }
      ion[i]=ion[i]->next_particle;
    }

    electron[i]=electron_start[i];
    while(electron[i]->next_particle!=NULL){
      if(myrank_x*Grid_Nx*dx+electron[i]->x<PROCESS_Nx*Grid_Nx*dx*0.8&&myrank_x*Grid_Nx*dx+electron[i]->x>PROCESS_Nx*Grid_Nx*dx*0.2&&
	 myrank_y*Grid_Ny*dx+electron[i]->y<PROCESS_Ny*Grid_Ny*dx*0.8&&myrank_y*Grid_Ny*dx+electron[i]->y>PROCESS_Ny*Grid_Ny*dx*0.2&&
	 electron[i]->z<Grid_Nz*dx*0.8&&electron[i]->z>Grid_Nz*dx*0.2){
	shape_func_v2(electron[i],&ex,&ey,&ez,&bx,&by,&bz);

	Vox=-(electron[i]->z+(-Grid_Nz/2-2)*dx)*omega_y-(electron[i]->y+(myrank_y*Grid_Ny-PROCESS_Ny*Grid_Ny/2-2)*dx)*omega_z;
	Voy=(electron[i]->x+(myrank_x*Grid_Nx-PROCESS_Nx*Grid_Nx/2-2)*dx)*omega_z;
	Voz=(electron[i]->x+(myrank_x*Grid_Nx-PROCESS_Nx*Grid_Nx/2-2)*dx)*omega_y;

	DIPOLE(
	       if(c<50000){
		 x=electron[i]->x+(myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
		 y=electron[i]->y+(myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
		 z=electron[i]->z+(-(Grid_Nz+4)/2)*dx;
		 
		 r=sqrt(x*x+y*y+z*z);
		 
		 bx+=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5))+IMF_x;
		 by+=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5))+IMF_y;
		 bz+=mu0*I*(1.+cos(w*c-Pi))/2.*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5))+IMF_z;
	       }else{
		 x=electron[i]->x+(myrank_x*Grid_Nx-(PROCESS_Nx*Grid_Nx+4)/2)*dx;
		 y=electron[i]->y+(myrank_y*Grid_Ny-(PROCESS_Ny*Grid_Ny+4)/2)*dx;
		 z=electron[i]->z+(-(Grid_Nz+4)/2)*dx;
		 
		 r=sqrt(x*x+y*y+z*z);
		 
		 bx+=mu0*I*Pi*R*R*(3.*x*y*sin(alpha)+3.*x*z*cos(alpha))/(4.*Pi*pow(r,5))+IMF_x;
		 by+=mu0*I*Pi*R*R*(3.*y*y*sin(alpha)+3.*y*z*cos(alpha)-r*r*sin(alpha))/(4.*Pi*pow(r,5))+IMF_y;
		 bz+=mu0*I*Pi*R*R*(3.*y*z*sin(alpha)+3.*z*z*cos(alpha)-r*r*cos(alpha))/(4.*Pi*pow(r,5))+IMF_z;
	       }
	       );

	F[0]-=q*(ex+(electron[i]->vy+Voy)*bz-(electron[i]->vz+Voz)*by)*electron[i]->n*dx*dx*dx;
	F[1]-=q*(ey+(electron[i]->vz+Voz)*bx-(electron[i]->vx+Vox)*bz)*electron[i]->n*dx*dx*dx;
	F[2]-=q*(ez+(electron[i]->vx+Vox)*by-(electron[i]->vy+Voy)*bx)*electron[i]->n*dx*dx*dx;
	
	if(electron[i]->flag>0){
	  F[9]-=q*ex*electron[i]->n*dx*dx*dx;
	  F[10]-=q*ey*electron[i]->n*dx*dx*dx;
	  F[11]-=q*ez*electron[i]->n*dx*dx*dx;
	  F[12]-=q*((electron[i]->vy+Voy)*bz-(electron[i]->vz+Voz)*by)*electron[i]->n*dx*dx*dx;
	  F[13]-=q*((electron[i]->vz+Voz)*bx-(electron[i]->vx+Vox)*bz)*electron[i]->n*dx*dx*dx;
	  F[14]-=q*((electron[i]->vx+Vox)*by-(electron[i]->vy+Voy)*bx)*electron[i]->n*dx*dx*dx;
	}else if(electron[i]->flag<0){
	  F[21]-=q*ex*electron[i]->n*dx*dx*dx;
	  F[22]-=q*ey*electron[i]->n*dx*dx*dx;
	  F[23]-=q*ez*electron[i]->n*dx*dx*dx;
	  F[24]-=q*((electron[i]->vy+Voy)*bz-(electron[i]->vz+Voz)*by)*electron[i]->n*dx*dx*dx;
	  F[25]-=q*((electron[i]->vz+Voz)*bx-(electron[i]->vx+Vox)*bz)*electron[i]->n*dx*dx*dx;
	  F[26]-=q*((electron[i]->vx+Vox)*by-(electron[i]->vy+Voy)*bx)*electron[i]->n*dx*dx*dx;
	}
      }
      electron[i]=electron[i]->next_particle;
    }
  }

  MPI_Allreduce(F,F_r,27,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  if(myrank_x==0&&myrank_y==0){
    for(i=0;i<27;i++){
      Fave[i]+=F_r[i]/500.;
    }

    if(c%500==0){

      sprintf(filename,"thrust%d_ave.txt",version);
      fp1=fopen(filename,"a");
      
      fprintf(fp1,"%d %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E\n",c,Fave[0],Fave[1],Fave[2],Fave[3],Fave[4],Fave[5],Fave[6],Fave[7],Fave[8],Fave[9],Fave[10],Fave[11],Fave[12],Fave[13],Fave[14],Fave[15],Fave[16],Fave[17],Fave[18],Fave[19],Fave[20],Fave[21],Fave[22],Fave[23],Fave[24],Fave[25],Fave[26]);
      fclose(fp1);

      for(i=0;i<27;i++){
	Fave[i]=0.;
      }
    }
  }

  return(0);
}

int add_ion(Particle *ion[],Particle *ion_start[],const int myrank_x,const int myrank_y)
{
  int i,j;
  int flag=0;
  Particle *p;
  double mu[3];
  double sigma0[3];
  double rho[3];
  double *vt;

  mu[0]=0.;
  mu[1]=0.;
  mu[2]=V;

  sigma0[0]=sqrt(kb*Ti/mi);
  sigma0[1]=sqrt(kb*Ti/mi);
  sigma0[2]=sqrt(kb*Ti/mi);

  rho[0]=0.;
  rho[1]=0.;
  rho[2]=0.;

#pragma omp parallel for  
  for(i=0;i<THREAD_N;i++){
    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){
      if((myrank_x==0&&ion[i]->x<4*dx)||(myrank_x==PROCESS_Nx-1&&ion[i]->x>Grid_Nx*dx)||
	 (myrank_y==0&&ion[i]->y<4*dx)||(myrank_y==PROCESS_Ny-1&&ion[i]->y>Grid_Ny*dx)||
	 ion[i]->z<4*dx||ion[i]->z>Grid_Nz*dx){
	del_particle_i(ion,i,ion_start);
      }
      ion[i]=ion[i]->next_particle;
    }
    if((myrank_x==0&&ion[i]->x<4*dx)||(myrank_x==PROCESS_Nx-1&&ion[i]->x>Grid_Nx*dx)||
       (myrank_y==0&&ion[i]->y<4*dx)||(myrank_y==PROCESS_Ny-1&&ion[i]->y>Grid_Ny*dx)||
       ion[i]->z<4*dx||ion[i]->z>Grid_Nz*dx){
      del_particle_i(ion,i,ion_start);
    }
  }
  
  //add xy plane
#pragma omp parallel for private(p,j)
  for(i=0;i<THREAD_N;i++){
    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){
      ion[i]=ion[i]->next_particle;
    }
    
    for(j=0;j<(Grid_Nx*Grid_Ny*Np*2)/THREAD_N;j++){
      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	p->x=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Nx+2)*dx);
	p->y=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Ny+2)*dx);
	p->z=gsl_ran_flat(rnd_p[i],2*dx,4*dx);

	vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	p->vx=vt[0];
	p->vy=vt[1];
	p->vz=vt[2];
	free(vt);

	p->n=N0i;
	//p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	p->flag=1;
	particle_flag[i]++;
	  
	ion[i]->next_particle=p;
	ion[i]->next_particle->prev_particle=ion[i];
	ion[i]=ion[i]->next_particle;
	ion[i]->next_particle=NULL;
      }else{
	fprintf(stderr,"Can't allocalte memory\n");
	exit(0);
      }
    }
    for(j=0;j<(Grid_Nx*Grid_Ny*Np*2)/THREAD_N;j++){
      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	p->x=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Nx+2)*dx);
	p->y=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Ny+2)*dx);
	p->z=gsl_ran_flat(rnd_p[i],(Grid_Nz)*dx,(Grid_Nz+2)*dx);

	vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	p->vx=vt[0];
	p->vy=vt[1];
	p->vz=vt[2];
	free(vt);

	p->n=N0i;
	//p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	p->flag=1;
	particle_flag[i]++;
	  
	ion[i]->next_particle=p;
	ion[i]->next_particle->prev_particle=ion[i];
	ion[i]=ion[i]->next_particle;
	ion[i]->next_particle=NULL;
      }else{
	fprintf(stderr,"Can't allocalte memory\n");
	exit(0);
      }
    }
  }

  //add xz plane
  if(myrank_y==0){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      for(j=0;j<(Grid_Nx*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Nx+2)*dx);
	  p->y=gsl_ran_flat(rnd_p[i],2*dx,4*dx);
	  p->z=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Nz)*dx);

	  vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];
	  free(vt);

	  p->n=N0i;
	  //p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	  p->flag=1;
	  particle_flag[i]++;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }
    }
  }

  if(myrank_y==PROCESS_Ny-1){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      for(j=0;j<(Grid_Nx*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Nx+2)*dx);
	  p->y=gsl_ran_flat(rnd_p[i],(Grid_Ny)*dx,(Grid_Ny+2)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Nz)*dx);

	  vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];
	  free(vt);

	  p->n=N0i;
	  //p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	  p->flag=1;
	  particle_flag[i]++;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }
    }
  }

  //add x direction
  if(myrank_x==0&&myrank_y==0){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      for(j=0;j<((Grid_Ny-2)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],2*dx,4*dx);
	  p->y=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Ny+2)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Nz)*dx);

	  vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];
	  free(vt);

	  p->n=N0i;
	  //p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	  p->flag=1;
	  particle_flag[i]++;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }
    }
  }else if(myrank_x==0&&myrank_y==PROCESS_Ny-1){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      for(j=0;j<((Grid_Ny-2)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],2*dx,4*dx);
	  p->y=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Ny)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Nz)*dx);

	  vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];
	  free(vt);

	  p->n=N0i;
	  //p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	  p->flag=1;
	  particle_flag[i]++;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }
    }
  }else if(myrank_x==PROCESS_Nx-1&&myrank_y==0){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      for(j=0;j<((Grid_Ny-2)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],(Grid_Nx)*dx,(Grid_Nx+2)*dx);
	  p->y=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Ny+2)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Nz)*dx);

	  vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];
	  free(vt);

	  p->n=N0i;
	  //p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	  p->flag=1;
	  particle_flag[i]++;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }
    }
  }else if(myrank_x==PROCESS_Nx-1&&myrank_y==PROCESS_Ny-1){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      for(j=0;j<((Grid_Ny-2)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],(Grid_Nx)*dx,(Grid_Nx+2)*dx);
	  p->y=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Ny)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Nz)*dx);

	  vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];
	  free(vt);

	  p->n=N0i;
	  //p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	  p->flag=1;
	  particle_flag[i]++;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }
    }
  }else if(myrank_x==0){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      for(j=0;j<((Grid_Ny)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],2*dx,4*dx);
	  p->y=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Ny+2)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Nz)*dx);

	  vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];
	  free(vt);

	  p->n=N0i;
	  //p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	  p->flag=1;
	  particle_flag[i]++;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }
    }
  }else if(myrank_x==PROCESS_Nx-1){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      for(j=0;j<((Grid_Ny)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],(Grid_Nx)*dx,(Grid_Nx+2)*dx);
	  p->y=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Ny+2)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Nz)*dx);

	  vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];
	  free(vt);

	  p->n=N0i;
	  //p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	  p->flag=1;
	  particle_flag[i]++;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }
    }
  }

  return(0);
}

int add_electron(Particle *electron[],Particle *electron_start[],const int myrank_x,const int myrank_y)
{
  int i,j;
  int flag=0;
  Particle *p;
  double mu[3];
  double sigma0[3];
  double rho[3];
  double *vt;

  mu[0]=0.;
  mu[1]=0.;
  mu[2]=V;

  sigma0[0]=sqrt(kb*Te/me);
  sigma0[1]=sqrt(kb*Te/me);
  sigma0[2]=sqrt(kb*Te/me);

  rho[0]=0.;
  rho[1]=0.;
  rho[2]=0.;

#pragma omp parallel for  
  for(i=0;i<THREAD_N;i++){
    electron[i]=electron_start[i];
    while(electron[i]->next_particle!=NULL){
      if((myrank_x==0&&electron[i]->x<4*dx)||(myrank_x==PROCESS_Nx-1&&electron[i]->x>Grid_Nx*dx)||
	 (myrank_y==0&&electron[i]->y<4*dx)||(myrank_y==PROCESS_Ny-1&&electron[i]->y>Grid_Ny*dx)||
	 electron[i]->z<4*dx||electron[i]->z>Grid_Nz*dx){
	del_particle_e(electron,i,electron_start);
      }
      electron[i]=electron[i]->next_particle;
    }
    if((myrank_x==0&&electron[i]->x<4*dx)||(myrank_x==PROCESS_Nx-1&&electron[i]->x>Grid_Nx*dx)||
       (myrank_y==0&&electron[i]->y<4*dx)||(myrank_y==PROCESS_Ny-1&&electron[i]->y>Grid_Ny*dx)||
       electron[i]->z<4*dx||electron[i]->z>Grid_Nz*dx){
      del_particle_e(electron,i,electron_start);
    }
  }
  
  //add xy plane
#pragma omp parallel for private(p,j)
  for(i=0;i<THREAD_N;i++){
    electron[i]=electron_start[i];
    while(electron[i]->next_particle!=NULL){
      electron[i]=electron[i]->next_particle;
    }
    
    for(j=0;j<(Grid_Nx*Grid_Ny*Np*2)/THREAD_N;j++){
      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	p->x=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Nx+2)*dx);
	p->y=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Ny+2)*dx);
	p->z=gsl_ran_flat(rnd_p[i],2*dx,4*dx);

	vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	p->vx=vt[0];
	p->vy=vt[1];
	p->vz=vt[2];
	free(vt);

	p->n=N0e;
	//p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	p->flag=1;
	particle_flag[i]++;
	  
	electron[i]->next_particle=p;
	electron[i]->next_particle->prev_particle=electron[i];
	electron[i]=electron[i]->next_particle;
	electron[i]->next_particle=NULL;
      }else{
	fprintf(stderr,"Can't allocalte memory\n");
	exit(0);
      }
    }
    for(j=0;j<(Grid_Nx*Grid_Ny*Np*2)/THREAD_N;j++){
      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	p->x=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Nx+2)*dx);
	p->y=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Ny+2)*dx);
	p->z=gsl_ran_flat(rnd_p[i],(Grid_Nz)*dx,(Grid_Nz+2)*dx);

	vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	p->vx=vt[0];
	p->vy=vt[1];
	p->vz=vt[2];
	free(vt);

	p->n=N0e;
	//p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	p->flag=1;
	particle_flag[i]++;
	  
	electron[i]->next_particle=p;
	electron[i]->next_particle->prev_particle=electron[i];
	electron[i]=electron[i]->next_particle;
	electron[i]->next_particle=NULL;
      }else{
	fprintf(stderr,"Can't allocalte memory\n");
	exit(0);
      }
    }
  }

  //add xz plane
  if(myrank_y==0){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      for(j=0;j<(Grid_Nx*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Nx+2)*dx);
	  p->y=gsl_ran_flat(rnd_p[i],2*dx,4*dx);
	  p->z=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Nz)*dx);

	  vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];
	  free(vt);

	  p->n=N0e;
	  //p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	  p->flag=1;
	  particle_flag[i]++;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }
    }
  }

  if(myrank_y==PROCESS_Ny-1){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      for(j=0;j<(Grid_Nx*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Nx+2)*dx);
	  p->y=gsl_ran_flat(rnd_p[i],(Grid_Ny)*dx,(Grid_Ny+2)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Nz)*dx);

	  vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];
	  free(vt);

	  p->n=N0e;
	  //p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	  p->flag=1;
	  particle_flag[i]++;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }
    }
  }

  //add yz plane
  if(myrank_x==0&&myrank_y==0){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      for(j=0;j<((Grid_Ny-2)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],2*dx,4*dx);
	  p->y=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Ny+2)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Nz)*dx);

	  vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];
	  free(vt);

	  p->n=N0e;
	  // p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	  p->flag=1;
	  particle_flag[i]++;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }
    }
  }else if(myrank_x==0&&myrank_y==PROCESS_Ny-1){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      for(j=0;j<((Grid_Ny-2)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],2*dx,4*dx);
	  p->y=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Ny)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Nz)*dx);

	  vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];
	  free(vt);

	  p->n=N0e;
	  //p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	  p->flag=1;
	  particle_flag[i]++;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }
    }
  }else if(myrank_x==PROCESS_Nx-1&&myrank_y==0){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      for(j=0;j<((Grid_Ny-2)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],(Grid_Nx)*dx,(Grid_Nx+2)*dx);
	  p->y=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Ny+2)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Nz)*dx);

	  vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];
	  free(vt);

	  p->n=N0e;
	  //p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	  p->flag=1;
	  particle_flag[i]++;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }
    }
  }else if(myrank_x==PROCESS_Nx-1&&myrank_y==PROCESS_Ny-1){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      for(j=0;j<((Grid_Ny-2)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],(Grid_Nx)*dx,(Grid_Nx+2)*dx);
	  p->y=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Ny)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Nz)*dx);

	  vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];
	  free(vt);

	  p->n=N0e;
	  //p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	  p->flag=1;
	  particle_flag[i]++;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }
    }
  }else if(myrank_x==0){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      for(j=0;j<((Grid_Ny)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],2*dx,4*dx);
	  p->y=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Ny+2)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Nz)*dx);

	  vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];
	  free(vt);

	  p->n=N0e;
	  //p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	  p->flag=1;
	  particle_flag[i]++;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }
    }
  }else if(myrank_x==PROCESS_Nx-1){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      for(j=0;j<((Grid_Ny)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],(Grid_Nx)*dx,(Grid_Nx+2)*dx);
	  p->y=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Ny+2)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],4*dx,(Grid_Nz)*dx);

	  vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];
	  free(vt);

	  p->n=N0e;
	  //p->flag=particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x);
	  p->flag=1;
	  particle_flag[i]++;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }
    }
  }

  return(0);
}

int add_ion_flat(Particle *ion[],Particle *ion_start[],const int myrank_x,const int myrank_y)
{
  int i,j,k,l,m;
  int flag=0;
  Particle *p;
  double mu[3];
  double sigma0[3];
  double rho[3];
  double *vt;

  mu[0]=0.;
  mu[1]=0.;
  mu[2]=V;

  sigma0[0]=sqrt(kb*Ti/mi);
  sigma0[1]=sqrt(kb*Ti/mi);
  sigma0[2]=sqrt(kb*Ti/mi);

  rho[0]=0.;
  rho[1]=0.;
  rho[2]=0.;

#pragma omp parallel for  
  for(i=0;i<THREAD_N;i++){
    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){
      if((myrank_x==0&&ion[i]->x<4*dx)||(myrank_x==PROCESS_Nx-1&&ion[i]->x>Grid_Nx*dx)||
	 (myrank_y==0&&ion[i]->y<4*dx)||(myrank_y==PROCESS_Ny-1&&ion[i]->y>Grid_Ny*dx)||
	 ion[i]->z<4*dx||ion[i]->z>Grid_Nz*dx){
	del_particle_i(ion,i,ion_start);
      }
      ion[i]=ion[i]->next_particle;
    }
    if((myrank_x==0&&ion[i]->x<4*dx)||(myrank_x==PROCESS_Nx-1&&ion[i]->x>Grid_Nx*dx)||
       (myrank_y==0&&ion[i]->y<4*dx)||(myrank_y==PROCESS_Ny-1&&ion[i]->y>Grid_Ny*dx)||
       ion[i]->z<4*dx||ion[i]->z>Grid_Nz*dx){
      del_particle_i(ion,i,ion_start);
    }
  }
  
  //add xy plane
#pragma omp parallel for private(p,j)
  for(i=0;i<THREAD_N;i++){
    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){
      ion[i]=ion[i]->next_particle;
    }
    
    // for(j=0;j<(Grid_Nx*Grid_Ny*Np*2)/THREAD_N;j++){
    for(k=0;k<Grid_Nx;k++){
      for(l=0;l<Grid_Ny;l++){
	for(m=0;m<2;m++){
	  for(j=0;j<Np/THREAD_N;j++){
	    if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	      p->x=gsl_ran_flat(rnd_p[i],(k+2)*dx,(k+3)*dx);
	      p->y=gsl_ran_flat(rnd_p[i],(l+2)*dx,(l+3)*dx);
	      p->z=gsl_ran_flat(rnd_p[i],(m+2)*dx,(m+3)*dx);
	      
	      vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	      p->vx=vt[0];
	      p->vy=vt[1];
	      p->vz=vt[2];
	      free(vt);
	      
	      p->n=N0i;
	      p->flag=1;
	      particle_flag[i]++;
	      
	      ion[i]->next_particle=p;
	      ion[i]->next_particle->prev_particle=ion[i];
	      ion[i]=ion[i]->next_particle;
	      ion[i]->next_particle=NULL;
	    }else{
	      fprintf(stderr,"Can't allocalte memory\n");
	      exit(0);
	    }
	  }
	}
      }
    }
    //for(j=0;j<(Grid_Nx*Grid_Ny*Np*2)/THREAD_N;j++){
    for(k=0;k<Grid_Nx;k++){
      for(l=0;l<Grid_Ny;l++){
	for(m=0;m<2;m++){
	  for(j=0;j<Np/THREAD_N;j++){
	    if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	      p->x=gsl_ran_flat(rnd_p[i],(k+2)*dx,(k+3)*dx);
	      p->y=gsl_ran_flat(rnd_p[i],(l+2)*dx,(l+3)*dx);
	      p->z=gsl_ran_flat(rnd_p[i],(m+Grid_Nz)*dx,(m+1+Grid_Nz)*dx);

	      vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	      p->vx=vt[0];
	      p->vy=vt[1];
	      p->vz=vt[2];
	      free(vt);
	      
	      p->n=N0i;
	      p->flag=1;
	      particle_flag[i]++;
	      
	      ion[i]->next_particle=p;
	      ion[i]->next_particle->prev_particle=ion[i];
	      ion[i]=ion[i]->next_particle;
	      ion[i]->next_particle=NULL;
	    }else{
	      fprintf(stderr,"Can't allocalte memory\n");
	      exit(0);
	    }
	  }
	}
      }
    }
  }

  //add xz plane
  if(myrank_y==0){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      //for(j=0;j<(Grid_Nx*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
      for(k=0;k<Grid_Nx;k++){
	for(l=0;l<2;l++){
	  for(m=0;m<Grid_Nz-4;m++){
	    for(j=0;j<Np/THREAD_N;j++){
	      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
		p->x=gsl_ran_flat(rnd_p[i],(k+2)*dx,(k+3)*dx);
		p->y=gsl_ran_flat(rnd_p[i],(l+2)*dx,(l+3)*dx);
		p->z=gsl_ran_flat(rnd_p[i],(m+4)*dx,(m+5)*dx);
		
		vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
		p->vx=vt[0];
		p->vy=vt[1];
		p->vz=vt[2];
		free(vt);
		
		p->n=N0i;
		p->flag=1;
		particle_flag[i]++;
		
		ion[i]->next_particle=p;
		ion[i]->next_particle->prev_particle=ion[i];
		ion[i]=ion[i]->next_particle;
		ion[i]->next_particle=NULL;
	      }else{
		fprintf(stderr,"Can't allocalte memory\n");
		exit(0);
	      }
	    }
	  }
	}
      }
    }
  }

  if(myrank_y==PROCESS_Ny-1){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      //for(j=0;j<(Grid_Nx*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
      for(k=0;k<Grid_Nx;k++){
	for(l=0;l<2;l++){
	  for(m=0;m<Grid_Nz-4;m++){
	    for(j=0;j<Np/THREAD_N;j++){
	      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
		p->x=gsl_ran_flat(rnd_p[i],(k+2)*dx,(k+3)*dx);
		p->y=gsl_ran_flat(rnd_p[i],(l+Grid_Ny)*dx,(l+1+Grid_Ny)*dx);
		p->z=gsl_ran_flat(rnd_p[i],(m+4)*dx,(m+5)*dx);

		vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
		p->vx=vt[0];
		p->vy=vt[1];
		p->vz=vt[2];
		free(vt);
		
		p->n=N0i;
		p->flag=1;
		particle_flag[i]++;
	  
		ion[i]->next_particle=p;
		ion[i]->next_particle->prev_particle=ion[i];
		ion[i]=ion[i]->next_particle;
		ion[i]->next_particle=NULL;
	      }else{
		fprintf(stderr,"Can't allocalte memory\n");
		exit(0);
	      }
	    }
	  }
	}
      }
    }
  }

  //add x direction
  if(myrank_x==0&&myrank_y==0){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      //for(j=0;j<((Grid_Ny-2)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
      for(k=0;k<2;k++){
	for(l=0;l<Grid_Ny-2;l++){
	  for(m=0;m<Grid_Nz-4;m++){
	    for(j=0;j<Np/THREAD_N;j++){
	      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
		p->x=gsl_ran_flat(rnd_p[i],(k+2)*dx,(k+3)*dx);
		p->y=gsl_ran_flat(rnd_p[i],(l+4)*dx,(l+5)*dx);
		p->z=gsl_ran_flat(rnd_p[i],(m+4)*dx,(m+5)*dx);

		vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
		p->vx=vt[0];
		p->vy=vt[1];
		p->vz=vt[2];
		free(vt);
		
		p->n=N0i;
		p->flag=1;
		particle_flag[i]++;
		
		ion[i]->next_particle=p;
		ion[i]->next_particle->prev_particle=ion[i];
		ion[i]=ion[i]->next_particle;
		ion[i]->next_particle=NULL;
	      }else{
		fprintf(stderr,"Can't allocalte memory\n");
		exit(0);
	      }
	    }
	  }
	}
      }
    }
  }else if(myrank_x==0&&myrank_y==PROCESS_Ny-1){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      //for(j=0;j<((Grid_Ny-2)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
      for(k=0;k<2;k++){
	for(l=0;l<Grid_Ny-2;l++){
	  for(m=0;m<Grid_Nz-4;m++){
	    for(j=0;j<Np/THREAD_N;j++){
	      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
		p->x=gsl_ran_flat(rnd_p[i],(k+2)*dx,(k+3)*dx);
		p->y=gsl_ran_flat(rnd_p[i],(l+2)*dx,(l+3)*dx);
		p->z=gsl_ran_flat(rnd_p[i],(m+4)*dx,(m+5)*dx);

		vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
		p->vx=vt[0];
		p->vy=vt[1];
		p->vz=vt[2];
		free(vt);
		
		p->n=N0i;
		p->flag=1;
		particle_flag[i]++;
		
		ion[i]->next_particle=p;
		ion[i]->next_particle->prev_particle=ion[i];
		ion[i]=ion[i]->next_particle;
		ion[i]->next_particle=NULL;
	      }else{
		fprintf(stderr,"Can't allocalte memory\n");
		exit(0);
	      }
	    }
	  }
	}
      }
    }
  }else if(myrank_x==PROCESS_Nx-1&&myrank_y==0){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      //for(j=0;j<((Grid_Ny-2)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
      for(k=0;k<2;k++){
	for(l=0;l<Grid_Ny-2;l++){
	  for(m=0;m<Grid_Nz-4;m++){
	    for(j=0;j<Np/THREAD_N;j++){
	      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
		p->x=gsl_ran_flat(rnd_p[i],(k+Grid_Nx)*dx,(k+1+Grid_Nx)*dx);
		p->y=gsl_ran_flat(rnd_p[i],(l+4)*dx,(l+5)*dx);
		p->z=gsl_ran_flat(rnd_p[i],(m+4)*dx,(m+5)*dx);

		vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
		p->vx=vt[0];
		p->vy=vt[1];
		p->vz=vt[2];
		free(vt);
		
		p->n=N0i;
		p->flag=1;
		particle_flag[i]++;
		
		ion[i]->next_particle=p;
		ion[i]->next_particle->prev_particle=ion[i];
		ion[i]=ion[i]->next_particle;
		ion[i]->next_particle=NULL;
	      }else{
		fprintf(stderr,"Can't allocalte memory\n");
		exit(0);
	      }
	    }
	  }
	}
      }
    }
  }else if(myrank_x==PROCESS_Nx-1&&myrank_y==PROCESS_Ny-1){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      //for(j=0;j<((Grid_Ny-2)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
      for(k=0;k<2;k++){
	for(l=0;l<Grid_Ny-2;l++){
	  for(m=0;m<Grid_Nz-4;m++){
	    for(j=0;j<Np/THREAD_N;j++){
	      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
		p->x=gsl_ran_flat(rnd_p[i],(k+Grid_Nx)*dx,(k+1+Grid_Nx)*dx);
		p->y=gsl_ran_flat(rnd_p[i],(l+2)*dx,(l+3)*dx);
		p->z=gsl_ran_flat(rnd_p[i],(m+4)*dx,(m+5)*dx);

		vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
		p->vx=vt[0];
		p->vy=vt[1];
		p->vz=vt[2];
		free(vt);
		
		p->n=N0i;
		p->flag=1;
		particle_flag[i]++;
		
		ion[i]->next_particle=p;
		ion[i]->next_particle->prev_particle=ion[i];
		ion[i]=ion[i]->next_particle;
		ion[i]->next_particle=NULL;
	      }else{
		fprintf(stderr,"Can't allocalte memory\n");
		exit(0);
	      }
	    }
	  }
	}
      }
    }
  }else if(myrank_x==0){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      //for(j=0;j<((Grid_Ny)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
      for(k=0;k<2;k++){
	for(l=0;l<Grid_Ny;l++){
	  for(m=0;m<Grid_Nz-4;m++){
	    for(j=0;j<Np/THREAD_N;j++){
	      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
		p->x=gsl_ran_flat(rnd_p[i],(k+2)*dx,(k+3)*dx);
		p->y=gsl_ran_flat(rnd_p[i],(l+2)*dx,(l+3)*dx);
		p->z=gsl_ran_flat(rnd_p[i],(m+4)*dx,(m+5)*dx);

		vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
		p->vx=vt[0];
		p->vy=vt[1];
		p->vz=vt[2];
		free(vt);
		
		p->n=N0i;
		p->flag=1;
		particle_flag[i]++;
		
		ion[i]->next_particle=p;
		ion[i]->next_particle->prev_particle=ion[i];
		ion[i]=ion[i]->next_particle;
		ion[i]->next_particle=NULL;
	      }else{
		fprintf(stderr,"Can't allocalte memory\n");
		exit(0);
	      }
	    }
	  }
	}
      }
    }
  }else if(myrank_x==PROCESS_Nx-1){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      //for(j=0;j<((Grid_Ny)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
      for(k=0;k<2;k++){
	for(l=0;l<Grid_Ny;l++){
	  for(m=0;m<Grid_Nz-4;m++){
	    for(j=0;j<Np/THREAD_N;j++){
	      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
		p->x=gsl_ran_flat(rnd_p[i],(k+Grid_Nx)*dx,(k+1+Grid_Nx)*dx);
		p->y=gsl_ran_flat(rnd_p[i],(l+2)*dx,(l+3)*dx);
		p->z=gsl_ran_flat(rnd_p[i],(m+4)*dx,(m+5)*dx);

		vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
		p->vx=vt[0];
		p->vy=vt[1];
		p->vz=vt[2];
		free(vt);
		
		p->n=N0i;
		p->flag=1;
		particle_flag[i]++;
		
		ion[i]->next_particle=p;
		ion[i]->next_particle->prev_particle=ion[i];
		ion[i]=ion[i]->next_particle;
		ion[i]->next_particle=NULL;
	      }else{
		fprintf(stderr,"Can't allocalte memory\n");
		exit(0);
	      }
	    }
	  }
	}
      }
    }
  }

  return(0);
}

int add_electron_flat(Particle *electron[],Particle *electron_start[],const int myrank_x,const int myrank_y)
{
  int i,j,k,l,m;
  int flag=0;
  Particle *p;
  double mu[3];
  double sigma0[3];
  double rho[3];
  double *vt;

  mu[0]=0.;
  mu[1]=0.;
  mu[2]=V;

  sigma0[0]=sqrt(kb*Te/me);
  sigma0[1]=sqrt(kb*Te/me);
  sigma0[2]=sqrt(kb*Te/me);

  rho[0]=0.;
  rho[1]=0.;
  rho[2]=0.;

#pragma omp parallel for  
  for(i=0;i<THREAD_N;i++){
    electron[i]=electron_start[i];
    while(electron[i]->next_particle!=NULL){
      if((myrank_x==0&&electron[i]->x<4*dx)||(myrank_x==PROCESS_Nx-1&&electron[i]->x>Grid_Nx*dx)||
	 (myrank_y==0&&electron[i]->y<4*dx)||(myrank_y==PROCESS_Ny-1&&electron[i]->y>Grid_Ny*dx)||
	 electron[i]->z<4*dx||electron[i]->z>Grid_Nz*dx){
	del_particle_e(electron,i,electron_start);
      }
      electron[i]=electron[i]->next_particle;
    }
    if((myrank_x==0&&electron[i]->x<4*dx)||(myrank_x==PROCESS_Nx-1&&electron[i]->x>Grid_Nx*dx)||
       (myrank_y==0&&electron[i]->y<4*dx)||(myrank_y==PROCESS_Ny-1&&electron[i]->y>Grid_Ny*dx)||
       electron[i]->z<4*dx||electron[i]->z>Grid_Nz*dx){
      del_particle_e(electron,i,electron_start);
    }
  }
  
  //add xy plane
#pragma omp parallel for private(p,k,l,m,j)
  for(i=0;i<THREAD_N;i++){
    electron[i]=electron_start[i];
    while(electron[i]->next_particle!=NULL){
      electron[i]=electron[i]->next_particle;
    }
    
    // for(j=0;j<(Grid_Nx*Grid_Ny*Np*2)/THREAD_N;j++){
    for(k=0;k<Grid_Nx;k++){
      for(l=0;l<Grid_Ny;l++){
	for(m=0;m<2;m++){
	  for(j=0;j<Np/THREAD_N;j++){
	    if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	      p->x=gsl_ran_flat(rnd_p[i],(k+2)*dx,(k+3)*dx);
	      p->y=gsl_ran_flat(rnd_p[i],(l+2)*dx,(l+3)*dx);
	      p->z=gsl_ran_flat(rnd_p[i],(m+2)*dx,(m+3)*dx);
	      
	      vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	      p->vx=vt[0];
	      p->vy=vt[1];
	      p->vz=vt[2];
	      free(vt);
	      
	      p->n=N0e;
	      p->flag=1;
	      particle_flag[i]++;
	      
	      electron[i]->next_particle=p;
	      electron[i]->next_particle->prev_particle=electron[i];
	      electron[i]=electron[i]->next_particle;
	      electron[i]->next_particle=NULL;
	    }else{
	      fprintf(stderr,"Can't allocalte memory\n");
	      exit(0);
	    }
	  }
	}
      }
    }
    //for(j=0;j<(Grid_Nx*Grid_Ny*Np*2)/THREAD_N;j++){
    for(k=0;k<Grid_Nx;k++){
      for(l=0;l<Grid_Ny;l++){
	for(m=0;m<2;m++){
	  for(j=0;j<Np/THREAD_N;j++){
	    if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	      p->x=gsl_ran_flat(rnd_p[i],(k+2)*dx,(k+3)*dx);
	      p->y=gsl_ran_flat(rnd_p[i],(l+2)*dx,(l+3)*dx);
	      p->z=gsl_ran_flat(rnd_p[i],(m+Grid_Nz)*dx,(m+1+Grid_Nz)*dx);

	      vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
	      p->vx=vt[0];
	      p->vy=vt[1];
	      p->vz=vt[2];
	      free(vt);
	      
	      p->n=N0e;
	      p->flag=1;
	      particle_flag[i]++;
	      
	      electron[i]->next_particle=p;
	      electron[i]->next_particle->prev_particle=electron[i];
	      electron[i]=electron[i]->next_particle;
	      electron[i]->next_particle=NULL;
	    }else{
	      fprintf(stderr,"Can't allocalte memory\n");
	      exit(0);
	    }
	  }
	}
      }
    }
  }

  //add xz plane
  if(myrank_y==0){
#pragma omp parallel for private(p,k,l,m,j)
    for(i=0;i<THREAD_N;i++){
      //for(j=0;j<(Grid_Nx*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
      for(k=0;k<Grid_Nx;k++){
	for(l=0;l<2;l++){
	  for(m=0;m<Grid_Nz-4;m++){
	    for(j=0;j<Np/THREAD_N;j++){
	      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
		p->x=gsl_ran_flat(rnd_p[i],(k+2)*dx,(k+3)*dx);
		p->y=gsl_ran_flat(rnd_p[i],(l+2)*dx,(l+3)*dx);
		p->z=gsl_ran_flat(rnd_p[i],(m+4)*dx,(m+5)*dx);
		
		vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
		p->vx=vt[0];
		p->vy=vt[1];
		p->vz=vt[2];
		free(vt);
		
		p->n=N0e;
		p->flag=1;
		particle_flag[i]++;
		
		electron[i]->next_particle=p;
		electron[i]->next_particle->prev_particle=electron[i];
		electron[i]=electron[i]->next_particle;
		electron[i]->next_particle=NULL;
	      }else{
		fprintf(stderr,"Can't allocalte memory\n");
		exit(0);
	      }
	    }
	  }
	}
      }
    }
  }

  if(myrank_y==PROCESS_Ny-1){
#pragma omp parallel for private(p,k,l,m,j)
    for(i=0;i<THREAD_N;i++){
      //for(j=0;j<(Grid_Nx*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
      for(k=0;k<Grid_Nx;k++){
	for(l=0;l<2;l++){
	  for(m=0;m<Grid_Nz-4;m++){
	    for(j=0;j<Np/THREAD_N;j++){
	      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
		p->x=gsl_ran_flat(rnd_p[i],(k+2)*dx,(k+3)*dx);
		p->y=gsl_ran_flat(rnd_p[i],(l+Grid_Ny)*dx,(l+1+Grid_Ny)*dx);
		p->z=gsl_ran_flat(rnd_p[i],(m+4)*dx,(m+5)*dx);

		vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
		p->vx=vt[0];
		p->vy=vt[1];
		p->vz=vt[2];
		free(vt);
		
		p->n=N0e;
		p->flag=1;
		particle_flag[i]++;
	  
		electron[i]->next_particle=p;
		electron[i]->next_particle->prev_particle=electron[i];
		electron[i]=electron[i]->next_particle;
		electron[i]->next_particle=NULL;
	      }else{
		fprintf(stderr,"Can't allocalte memory\n");
		exit(0);
	      }
	    }
	  }
	}
      }
    }
  }

  //add x directelectron
  if(myrank_x==0&&myrank_y==0){
#pragma omp parallel for private(p,k,l,m,j)
    for(i=0;i<THREAD_N;i++){
      //for(j=0;j<((Grid_Ny-2)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
      for(k=0;k<2;k++){
	for(l=0;l<Grid_Ny-2;l++){
	  for(m=0;m<Grid_Nz-4;m++){
	    for(j=0;j<Np/THREAD_N;j++){
	      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
		p->x=gsl_ran_flat(rnd_p[i],(k+2)*dx,(k+3)*dx);
		p->y=gsl_ran_flat(rnd_p[i],(l+4)*dx,(l+5)*dx);
		p->z=gsl_ran_flat(rnd_p[i],(m+4)*dx,(m+5)*dx);

		vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
		p->vx=vt[0];
		p->vy=vt[1];
		p->vz=vt[2];
		free(vt);
		
		p->n=N0e;
		p->flag=1;
		particle_flag[i]++;
		
		electron[i]->next_particle=p;
		electron[i]->next_particle->prev_particle=electron[i];
		electron[i]=electron[i]->next_particle;
		electron[i]->next_particle=NULL;
	      }else{
		fprintf(stderr,"Can't allocalte memory\n");
		exit(0);
	      }
	    }
	  }
	}
      }
    }
  }else if(myrank_x==0&&myrank_y==PROCESS_Ny-1){
#pragma omp parallel for private(p,k,l,m,j)
    for(i=0;i<THREAD_N;i++){
      //for(j=0;j<((Grid_Ny-2)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
      for(k=0;k<2;k++){
	for(l=0;l<Grid_Ny-2;l++){
	  for(m=0;m<Grid_Nz-4;m++){
	    for(j=0;j<Np/THREAD_N;j++){
	      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
		p->x=gsl_ran_flat(rnd_p[i],(k+2)*dx,(k+3)*dx);
		p->y=gsl_ran_flat(rnd_p[i],(l+2)*dx,(l+3)*dx);
		p->z=gsl_ran_flat(rnd_p[i],(m+4)*dx,(m+5)*dx);

		vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
		p->vx=vt[0];
		p->vy=vt[1];
		p->vz=vt[2];
		free(vt);
		
		p->n=N0e;
		p->flag=1;
		particle_flag[i]++;
		
		electron[i]->next_particle=p;
		electron[i]->next_particle->prev_particle=electron[i];
		electron[i]=electron[i]->next_particle;
		electron[i]->next_particle=NULL;
	      }else{
		fprintf(stderr,"Can't allocalte memory\n");
		exit(0);
	      }
	    }
	  }
	}
      }
    }
  }else if(myrank_x==PROCESS_Nx-1&&myrank_y==0){
#pragma omp parallel for private(p,k,l,m,j)
    for(i=0;i<THREAD_N;i++){
      //for(j=0;j<((Grid_Ny-2)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
      for(k=0;k<2;k++){
	for(l=0;l<Grid_Ny-2;l++){
	  for(m=0;m<Grid_Nz-4;m++){
	    for(j=0;j<Np/THREAD_N;j++){
	      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
		p->x=gsl_ran_flat(rnd_p[i],(k+Grid_Nx)*dx,(k+1+Grid_Nx)*dx);
		p->y=gsl_ran_flat(rnd_p[i],(l+4)*dx,(l+5)*dx);
		p->z=gsl_ran_flat(rnd_p[i],(m+4)*dx,(m+5)*dx);

		vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
		p->vx=vt[0];
		p->vy=vt[1];
		p->vz=vt[2];
		free(vt);
		
		p->n=N0e;
		p->flag=1;
		particle_flag[i]++;
		
		electron[i]->next_particle=p;
		electron[i]->next_particle->prev_particle=electron[i];
		electron[i]=electron[i]->next_particle;
		electron[i]->next_particle=NULL;
	      }else{
		fprintf(stderr,"Can't allocalte memory\n");
		exit(0);
	      }
	    }
	  }
	}
      }
    }
  }else if(myrank_x==PROCESS_Nx-1&&myrank_y==PROCESS_Ny-1){
#pragma omp parallel for private(p,k,l,m,j)
    for(i=0;i<THREAD_N;i++){
      //for(j=0;j<((Grid_Ny-2)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
      for(k=0;k<2;k++){
	for(l=0;l<Grid_Ny-2;l++){
	  for(m=0;m<Grid_Nz-4;m++){
	    for(j=0;j<Np/THREAD_N;j++){
	      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
		p->x=gsl_ran_flat(rnd_p[i],(k+Grid_Nx)*dx,(k+1+Grid_Nx)*dx);
		p->y=gsl_ran_flat(rnd_p[i],(l+2)*dx,(l+3)*dx);
		p->z=gsl_ran_flat(rnd_p[i],(m+4)*dx,(m+5)*dx);

		vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
		p->vx=vt[0];
		p->vy=vt[1];
		p->vz=vt[2];
		free(vt);
		
		p->n=N0e;
		p->flag=1;
		particle_flag[i]++;
		
		electron[i]->next_particle=p;
		electron[i]->next_particle->prev_particle=electron[i];
		electron[i]=electron[i]->next_particle;
		electron[i]->next_particle=NULL;
	      }else{
		fprintf(stderr,"Can't allocalte memory\n");
		exit(0);
	      }
	    }
	  }
	}
      }
    }
  }else if(myrank_x==0){
#pragma omp parallel for private(p,k,l,m,j)
    for(i=0;i<THREAD_N;i++){
      //for(j=0;j<((Grid_Ny)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
      for(k=0;k<2;k++){
	for(l=0;l<Grid_Ny;l++){
	  for(m=0;m<Grid_Nz-4;m++){
	    for(j=0;j<Np/THREAD_N;j++){
	      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
		p->x=gsl_ran_flat(rnd_p[i],(k+2)*dx,(k+3)*dx);
		p->y=gsl_ran_flat(rnd_p[i],(l+2)*dx,(l+3)*dx);
		p->z=gsl_ran_flat(rnd_p[i],(m+4)*dx,(m+5)*dx);

		vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
		p->vx=vt[0];
		p->vy=vt[1];
		p->vz=vt[2];
		free(vt);
		
		p->n=N0e;
		p->flag=1;
		particle_flag[i]++;
		
		electron[i]->next_particle=p;
		electron[i]->next_particle->prev_particle=electron[i];
		electron[i]=electron[i]->next_particle;
		electron[i]->next_particle=NULL;
	      }else{
		fprintf(stderr,"Can't allocalte memory\n");
		exit(0);
	      }
	    }
	  }
	}
      }
    }
  }else if(myrank_x==PROCESS_Nx-1){
#pragma omp parallel for private(p,k,l,m,j)
    for(i=0;i<THREAD_N;i++){
      //for(j=0;j<((Grid_Ny)*(Grid_Nz-4)*Np*2)/THREAD_N;j++){
      for(k=0;k<2;k++){
	for(l=0;l<Grid_Ny;l++){
	  for(m=0;m<Grid_Nz-4;m++){
	    for(j=0;j<Np/THREAD_N;j++){
	      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
		p->x=gsl_ran_flat(rnd_p[i],(k+Grid_Nx)*dx,(k+1+Grid_Nx)*dx);
		p->y=gsl_ran_flat(rnd_p[i],(l+2)*dx,(l+3)*dx);
		p->z=gsl_ran_flat(rnd_p[i],(m+4)*dx,(m+5)*dx);

		vt=ran_multivar_gaussian(rnd_v[i],3,mu,sigma0,rho);
		p->vx=vt[0];
		p->vy=vt[1];
		p->vz=vt[2];
		free(vt);
		
		p->n=N0e;
		p->flag=1;
		particle_flag[i]++;
		
		electron[i]->next_particle=p;
		electron[i]->next_particle->prev_particle=electron[i];
		electron[i]=electron[i]->next_particle;
		electron[i]->next_particle=NULL;
	      }else{
		fprintf(stderr,"Can't allocalte memory\n");
		exit(0);
	      }
	    }
	  }
	}
      }
    }
  }

  return(0);
}

int inject_ion_para4(Particle *ion[],Particle *ion_start[],const int myrank_x,const int myrank_y)
{
  int i,j;
  int flag=0;
  const int inj_c=10000;
  Particle *p;
  double r,theta;
  const double Ri=300.;//125;//120.;
  const double Rz=75.;//15.;
  const double TTi=1.;
  const double Vr=1e6;
  double x0,y0,z0;
  double mu[3];
  double sigma0[3];
  double rho[3];
  double *vt;

  //#pragma omp parallel for private(p,j,r,theta,x0,y0)
  for(i=0;i<THREAD_N;i++){
    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){
      ion[i]=ion[i]->next_particle;
    }
    
    for(j=0;j<(int)(2*inj_c*Np)/THREAD_N;j++){
      //r=Ri*gsl_ran_flat(rnd_i[i],0.5,1.5);
      r=Ri*gsl_ran_flat(rnd_i[i],0.50,1.5);
      theta=gsl_ran_flat(rnd_i[i],0.,2.*Pi);
      x0=r*cos(theta)+Grid_Nx*PROCESS_Nx/2.*dx;
      y0=r*sin(theta)+Grid_Ny*PROCESS_Ny/2.*dx;
      z0=gsl_ran_flat(rnd_i[i],(Grid_Nz+4)/2*dx-0.5*dx,(Grid_Nz+4)/2*dx+0.5*dx)-Rz*gsl_ran_flat(rnd_i[i],-2.,2.);

      mu[0]=Vr*cos(theta/*-Pi/2.*/);
      mu[1]=Vr*sin(theta/*-Pi/2.*/);
      mu[2]=0.;
      
      sigma0[0]=sqrt(kb*TTi*Ti/mi);
      sigma0[1]=sqrt(kb*TTi*Ti/mi);
      sigma0[2]=sqrt(kb*TTi*Ti/mi);
      
      rho[0]=0.;
      rho[1]=0.;
      rho[2]=0.;
      
      vt=ran_multivar_gaussian(rnd_i[i],3,mu,sigma0,rho);
      while(vt[0]*vt[0]+vt[1]*vt[1]+vt[2]*vt[2]>=C*C){
	vt=ran_multivar_gaussian(rnd_i[i],3,mu,sigma0,rho);
      }

      if(myrank_x==(int)(x0/(Grid_Nx*dx))&&myrank_y==(int)(y0/(Grid_Ny*dx))){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=x0+2.*dx-Grid_Nx*myrank_x*dx;
	  p->y=y0+2.*dx-Grid_Ny*myrank_y*dx;
	  p->z=z0;

	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];

	  p->n=N1i;
	  p->flag=-1;//-(particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x));
	  particle_flag[i]++;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }

      theta+=Pi/2.;
      x0=r*cos(theta)+Grid_Nx*PROCESS_Nx/2.*dx;
      y0=r*sin(theta)+Grid_Ny*PROCESS_Ny/2.*dx;

      if(myrank_x==(int)(x0/(Grid_Nx*dx))&&myrank_y==(int)(y0/(Grid_Ny*dx))){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=x0+2.*dx-Grid_Nx*myrank_x*dx;
	  p->y=y0+2.*dx-Grid_Ny*myrank_y*dx;
	  p->z=z0;

	  p->vx=-vt[1];
	  p->vy=vt[0];
	  p->vz=vt[2];

	  p->n=N1i;
	  p->flag=-1;//-(particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x));
	  particle_flag[i]++;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }

      theta+=Pi/2.;
      x0=r*cos(theta)+Grid_Nx*PROCESS_Nx/2.*dx;
      y0=r*sin(theta)+Grid_Ny*PROCESS_Ny/2.*dx;

      if(myrank_x==(int)(x0/(Grid_Nx*dx))&&myrank_y==(int)(y0/(Grid_Ny*dx))){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=x0+2.*dx-Grid_Nx*myrank_x*dx;
	  p->y=y0+2.*dx-Grid_Ny*myrank_y*dx;
	  p->z=z0;

	  p->vx=-vt[0];
	  p->vy=-vt[1];
	  p->vz=vt[2];

	  p->n=N1i;
	  p->flag=-1;//-(particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x));
	  particle_flag[i]++;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }

      theta+=Pi/2.;
      x0=r*cos(theta)+Grid_Nx*PROCESS_Nx/2.*dx;
      y0=r*sin(theta)+Grid_Ny*PROCESS_Ny/2.*dx;

      if(myrank_x==(int)(x0/(Grid_Nx*dx))&&myrank_y==(int)(y0/(Grid_Ny*dx))){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=x0+2.*dx-Grid_Nx*myrank_x*dx;
	  p->y=y0+2.*dx-Grid_Ny*myrank_y*dx;
	  p->z=z0;

	  p->vx=vt[1];
	  p->vy=-vt[0];
	  p->vz=vt[2];

	  p->n=N1i;
	  p->flag=-1;//-(particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x));
	  particle_flag[i]++;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }

      free(vt);
    }
  }

  return(0);
}

int inject_electron_para4(Particle *electron[],Particle *electron_start[],const int myrank_x,const int myrank_y)
{
  int i,j;
  int flag=0;
  const int inj_c=10000;
  Particle *p;
  double r,theta;
  const double Re=300.;//120.;
  const double Rz=75.;//15.;
  const double TTe=1.;
  const double Vr=1e8;
  double x0,y0,z0;
  double mu[3];
  double sigma0[3];
  double rho[3];
  double *vt;

  //#pragma omp parallel for private(p,j,r,theta,x0,y0)
  for(i=0;i<THREAD_N;i++){
    electron[i]=electron_start[i];
    while(electron[i]->next_particle!=NULL){
      electron[i]=electron[i]->next_particle;
    }
    
    for(j=0;j<(int)(2*inj_c*Np)/THREAD_N;j++){
      //r=Re*gsl_ran_flat(rnd_i[i],0.5,1.5);
      r=Re*gsl_ran_flat(rnd_i[i],0.50,1.5);
      theta=gsl_ran_flat(rnd_i[i],0.,2.*Pi);
      x0=r*cos(theta)+Grid_Nx*PROCESS_Nx/2.*dx;
      y0=r*sin(theta)+Grid_Ny*PROCESS_Ny/2.*dx;
      z0=gsl_ran_flat(rnd_i[i],(Grid_Nz+4)/2*dx-0.5*dx,(Grid_Nz+4)/2*dx+0.5*dx)-Rz*gsl_ran_flat(rnd_i[i],-2.,2.);

      mu[0]=Vr*cos(theta/*-Pi/2.*/);
      mu[1]=Vr*sin(theta/*-Pi/2.*/);
      mu[2]=0.;
      
      sigma0[0]=sqrt(kb*TTe*Te/me);
      sigma0[1]=sqrt(kb*TTe*Te/me);
      sigma0[2]=sqrt(kb*TTe*Te/me);
      
      rho[0]=0.;
      rho[1]=0.;
      rho[2]=0.;
      
      vt=ran_multivar_gaussian(rnd_i[i],3,mu,sigma0,rho);
      while(vt[0]*vt[0]+vt[1]*vt[1]+vt[2]*vt[2]>=C*C){
	vt=ran_multivar_gaussian(rnd_i[i],3,mu,sigma0,rho);
      }

      if(myrank_x==(int)(x0/(Grid_Nx*dx))&&myrank_y==(int)(y0/(Grid_Ny*dx))){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=x0+2.*dx-Grid_Nx*myrank_x*dx;
	  p->y=y0+2.*dx-Grid_Ny*myrank_y*dx;
	  p->z=z0;

	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];

	  p->n=N1e;
	  p->flag=-1;//-(particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x));
	  particle_flag[i]++;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }

      theta+=Pi/2.;
      x0=r*cos(theta)+Grid_Nx*PROCESS_Nx/2.*dx;
      y0=r*sin(theta)+Grid_Ny*PROCESS_Ny/2.*dx;

      if(myrank_x==(int)(x0/(Grid_Nx*dx))&&myrank_y==(int)(y0/(Grid_Ny*dx))){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=x0+2.*dx-Grid_Nx*myrank_x*dx;
	  p->y=y0+2.*dx-Grid_Ny*myrank_y*dx;
	  p->z=z0;

	  p->vx=-vt[1];
	  p->vy=vt[0];
	  p->vz=vt[2];

	  p->n=N1e;
	  p->flag=-1;//-(particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x));
	  particle_flag[i]++;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }

      theta+=Pi/2.;
      x0=r*cos(theta)+Grid_Nx*PROCESS_Nx/2.*dx;
      y0=r*sin(theta)+Grid_Ny*PROCESS_Ny/2.*dx;

      if(myrank_x==(int)(x0/(Grid_Nx*dx))&&myrank_y==(int)(y0/(Grid_Ny*dx))){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=x0+2.*dx-Grid_Nx*myrank_x*dx;
	  p->y=y0+2.*dx-Grid_Ny*myrank_y*dx;
	  p->z=z0;

	  p->vx=-vt[0];
	  p->vy=-vt[1];
	  p->vz=vt[2];

	  p->n=N1e;
	  p->flag=-1;//-(particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x));
	  particle_flag[i]++;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }

      theta+=Pi/2.;
      x0=r*cos(theta)+Grid_Nx*PROCESS_Nx/2.*dx;
      y0=r*sin(theta)+Grid_Ny*PROCESS_Ny/2.*dx;

      if(myrank_x==(int)(x0/(Grid_Nx*dx))&&myrank_y==(int)(y0/(Grid_Ny*dx))){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=x0+2.*dx-Grid_Nx*myrank_x*dx;
	  p->y=y0+2.*dx-Grid_Ny*myrank_y*dx;
	  p->z=z0;

	  p->vx=vt[1];
	  p->vy=-vt[0];
	  p->vz=vt[2];

	  p->n=N1e;
	  p->flag=-1;//-(particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x));
	  particle_flag[i]++;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }

      free(vt);
    }
  }

  return(0);
}

int inject_ion_perp4(Particle *ion[],Particle *ion_start[],const int myrank_x,const int myrank_y)
{
  int i,j;
  int flag=0;
  const int inj_c=10000;
  Particle *p;
  double r,theta;
  const double Ri=300.;//125;//120.;
  const double Ry=75.;//15.;
  const double TTi=1.;
  const double Vr=1e6;
  double x0,y0,z0;
  double mu[3];
  double sigma0[3];
  double rho[3];
  double *vt;

  //#pragma omp parallel for private(p,j,r,theta,x0,y0)
  for(i=0;i<THREAD_N;i++){
    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){
      ion[i]=ion[i]->next_particle;
    }
    
    for(j=0;j<(int)(2*inj_c*Np)/THREAD_N;j++){
      r=Ri*gsl_ran_flat(rnd_i[i],0.5,1.5);
      theta=gsl_ran_flat(rnd_i[i],0.,2.*Pi);
      x0=r*cos(theta)+(Grid_Nx*PROCESS_Nx)/2.*dx;
      y0=gsl_ran_flat(rnd_i[i],(Grid_Ny*PROCESS_Ny)/2.*dx-0.5*dx,(Grid_Ny*PROCESS_Ny)/2*dx+0.5*dx)-Ry*gsl_ran_flat(rnd_i[i],-2.,2.);
      z0=r*sin(theta)+(Grid_Nz+4)/2.*dx;


      mu[0]=Vr*cos(theta/*-Pi/2.*/);
      mu[1]=0.;
      mu[2]=Vr*sin(theta/*-Pi/2.*/);
      
      sigma0[0]=sqrt(kb*TTi*Ti/mi);
      sigma0[1]=sqrt(kb*TTi*Ti/mi);
      sigma0[2]=sqrt(kb*TTi*Ti/mi);
      
      rho[0]=0.;
      rho[1]=0.;
      rho[2]=0.;
      
      vt=ran_multivar_gaussian(rnd_i[i],3,mu,sigma0,rho);
      while(vt[0]*vt[0]+vt[1]*vt[1]+vt[2]*vt[2]>=C*C){
	vt=ran_multivar_gaussian(rnd_i[i],3,mu,sigma0,rho);
      }

      if(myrank_x==(int)(x0/(Grid_Nx*dx))&&myrank_y==(int)(y0/(Grid_Ny*dx))){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=x0+2.*dx-Grid_Nx*myrank_x*dx;
	  p->y=y0+2.*dx-Grid_Ny*myrank_y*dx;
	  p->z=z0;

	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];

	  p->n=N1i;
	  p->flag=-1;//-(particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x));
	  particle_flag[i]++;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }

      theta+=Pi/2.;
      x0=r*cos(theta)+(Grid_Nx*PROCESS_Nx)/2.*dx;
      z0=r*sin(theta)+(Grid_Nz+4)/2.*dx;

      if(myrank_x==(int)(x0/(Grid_Nx*dx))&&myrank_y==(int)(y0/(Grid_Ny*dx))){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=x0+2.*dx-Grid_Nx*myrank_x*dx;
	  p->y=y0+2.*dx-Grid_Ny*myrank_y*dx;
	  p->z=z0;

	  p->vx=-vt[2];
	  p->vy=vt[1];
	  p->vz=vt[0];

	  p->n=N1i;
	  p->flag=-1;//-(particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x));
	  particle_flag[i]++;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }

      theta+=Pi/2.;
      x0=r*cos(theta)+(Grid_Nx*PROCESS_Nx)/2.*dx;
      z0=r*sin(theta)+(Grid_Nz+4)/2.*dx;

      if(myrank_x==(int)(x0/(Grid_Nx*dx))&&myrank_y==(int)(y0/(Grid_Ny*dx))){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=x0+2.*dx-Grid_Nx*myrank_x*dx;
	  p->y=y0+2.*dx-Grid_Ny*myrank_y*dx;
	  p->z=z0;

	  p->vx=-vt[0];
	  p->vy=vt[1];
	  p->vz=-vt[2];

	  p->n=N1i;
	  p->flag=-1;//-(particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x));
	  particle_flag[i]++;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }

      theta+=Pi/2.;
      x0=r*cos(theta)+(Grid_Nx*PROCESS_Nx)/2.*dx;
      z0=r*sin(theta)+(Grid_Nz+4)/2.*dx;

      if(myrank_x==(int)(x0/(Grid_Nx*dx))&&myrank_y==(int)(y0/(Grid_Ny*dx))){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=x0+2.*dx-Grid_Nx*myrank_x*dx;
	  p->y=y0+2.*dx-Grid_Ny*myrank_y*dx;
	  p->z=z0;

	  p->vx=vt[2];
	  p->vy=vt[1];
	  p->vz=-vt[0];

	  p->n=N1i;
	  p->flag=-1;//-(particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x));
	  particle_flag[i]++;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }

      free(vt);
    }
  }

  return(0);
}

int inject_electron_perp4(Particle *electron[],Particle *electron_start[],const int myrank_x,const int myrank_y)
{
  int i,j;
  int flag=0;
  const int inj_c=10000;
  Particle *p;
  double r,theta;
  const double Re=300.;//120.;
  const double Ry=75.;//15.;
  const double TTe=1.;
  const double Vr=1e8;
  double x0,y0,z0;
  double mu[3];
  double sigma0[3];
  double rho[3];
  double *vt;

  //#pragma omp parallel for private(p,j,r,theta,x0,y0)
  for(i=0;i<THREAD_N;i++){
    electron[i]=electron_start[i];
    while(electron[i]->next_particle!=NULL){
      electron[i]=electron[i]->next_particle;
    }
    
    for(j=0;j<(int)(2*inj_c*Np)/THREAD_N;j++){
      r=Re*gsl_ran_flat(rnd_i[i],0.5,1.5);
      theta=gsl_ran_flat(rnd_i[i],0.,2.*Pi);
      x0=r*cos(theta)+(Grid_Nx*PROCESS_Nx)/2.*dx;
      y0=gsl_ran_flat(rnd_i[i],(Grid_Ny*PROCESS_Ny)/2.*dx-0.5*dx,(Grid_Ny*PROCESS_Ny)/2*dx+0.5*dx)-Ry*gsl_ran_flat(rnd_i[i],-2.,2.);
      z0=r*sin(theta)+(Grid_Nz+4)/2.*dx;


      mu[0]=Vr*cos(theta/*-Pi/2.*/);
      mu[1]=0.;
      mu[2]=Vr*sin(theta/*-Pi/2.*/);
      
      sigma0[0]=sqrt(kb*TTe*Te/me);
      sigma0[1]=sqrt(kb*TTe*Te/me);
      sigma0[2]=sqrt(kb*TTe*Te/me);
      
      rho[0]=0.;
      rho[1]=0.;
      rho[2]=0.;
      
      vt=ran_multivar_gaussian(rnd_i[i],3,mu,sigma0,rho);
      while(vt[0]*vt[0]+vt[1]*vt[1]+vt[2]*vt[2]>=C*C){
	vt=ran_multivar_gaussian(rnd_i[i],3,mu,sigma0,rho);
      }

      if(myrank_x==(int)(x0/(Grid_Nx*dx))&&myrank_y==(int)(y0/(Grid_Ny*dx))){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=x0+2.*dx-Grid_Nx*myrank_x*dx;
	  p->y=y0+2.*dx-Grid_Ny*myrank_y*dx;
	  p->z=z0;

	  p->vx=vt[0];
	  p->vy=vt[1];
	  p->vz=vt[2];

	  p->n=N1e;
	  p->flag=-1;//-(particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x));
	  particle_flag[i]++;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }

      theta+=Pi/2.;
      x0=r*cos(theta)+(Grid_Nx*PROCESS_Nx)/2.*dx;
      z0=r*sin(theta)+(Grid_Nz+4)/2.*dx;

      if(myrank_x==(int)(x0/(Grid_Nx*dx))&&myrank_y==(int)(y0/(Grid_Ny*dx))){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=x0+2.*dx-Grid_Nx*myrank_x*dx;
	  p->y=y0+2.*dx-Grid_Ny*myrank_y*dx;
	  p->z=z0;

	  p->vx=-vt[2];
	  p->vy=vt[1];
	  p->vz=vt[0];

	  p->n=N1e;
	  p->flag=-1;//-(particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x));
	  particle_flag[i]++;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }

      theta+=Pi/2.;
      x0=r*cos(theta)+(Grid_Nx*PROCESS_Nx)/2.*dx;
      z0=r*sin(theta)+(Grid_Nz+4)/2.*dx;

      if(myrank_x==(int)(x0/(Grid_Nx*dx))&&myrank_y==(int)(y0/(Grid_Ny*dx))){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=x0+2.*dx-Grid_Nx*myrank_x*dx;
	  p->y=y0+2.*dx-Grid_Ny*myrank_y*dx;
	  p->z=z0;

	  p->vx=-vt[0];
	  p->vy=vt[1];
	  p->vz=-vt[2];

	  p->n=N1e;
	  p->flag=-1;//-(particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x));
	  particle_flag[i]++;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }

      theta+=Pi/2.;
      x0=r*cos(theta)+(Grid_Nx*PROCESS_Nx)/2.*dx;
      z0=r*sin(theta)+(Grid_Nz+4)/2.*dx;

      if(myrank_x==(int)(x0/(Grid_Nx*dx))&&myrank_y==(int)(y0/(Grid_Ny*dx))){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=x0+2.*dx-Grid_Nx*myrank_x*dx;
	  p->y=y0+2.*dx-Grid_Ny*myrank_y*dx;
	  p->z=z0;

	  p->vx=vt[2];
	  p->vy=vt[1];
	  p->vz=-vt[0];

	  p->n=N1e;
	  p->flag=-1;//-(particle_flag[i]*100000+(myrank_y*PROCESS_Nx+myrank_x));
	  particle_flag[i]++;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }

      free(vt);
    }
  }

  return(0);
}

int sorting(Particle *particle[],Particle *particle_start[])
{
  int k,l,m,i;
  int count[THREAD_N][Grid_Nx][Grid_Ny][Grid_Nz];

#pragma omp parallel for private(k,l,m)
  for(i=0;i<THREAD_N;i++){
    particle[i]=particle_start[i];
    for(k=0;k<Grid_Nx;k++){
      for(l=0;l<Grid_Ny;l++){
	for(m=0;m<Grid_Nz;m++){
	  sort_particle[i][k][l][m]=sort_particle_start[i][k][l][m];
	  count[i][k][l][m]=0;
	}
      }
    }
    
    while(particle[i]->next_particle!=NULL){
      k=(int)(particle[i]->x/dx)-2;
      l=(int)(particle[i]->y/dx)-2;
      m=(int)(particle[i]->z/dx)-2;

      if(k>=0&&k<Grid_Nx&&l>=0&&l<Grid_Ny&&m>=0&&m<Grid_Nz){
	count[i][k][l][m]++;
	if((sort_particle[i][k][l][m]->next_particle=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  memcpy(sort_particle[i][k][l][m]->next_particle,particle[i],sizeof(Particle));
	  sort_particle[i][k][l][m]->next_particle->next_particle=NULL;
	  sort_particle[i][k][l][m]->next_particle->prev_particle=sort_particle[i][k][l][m];
	  sort_particle[i][k][l][m]=sort_particle[i][k][l][m]->next_particle;
	}else{
	  fprintf(stderr,"Can't allocalte memory\n");
	  exit(0);
	}
      }

      del_particle_i(particle,i,particle_start);
      particle[i]=particle[i]->next_particle;
    }

    k=(int)(particle[i]->x/dx)-2;
    l=(int)(particle[i]->y/dx)-2;
    m=(int)(particle[i]->z/dx)-2;
    
    if(k>=0&&k<Grid_Nx&&l>=0&&l<Grid_Ny&&m>=0&&m<Grid_Nz){
      count[i][k][l][m]++;
      if((sort_particle[i][k][l][m]->next_particle=(Particle*)malloc(sizeof(Particle)))!=NULL){
	memcpy(sort_particle[i][k][l][m]->next_particle,particle[i],sizeof(Particle));
	sort_particle[i][k][l][m]->next_particle->next_particle=NULL;
	sort_particle[i][k][l][m]->next_particle->prev_particle=sort_particle[i][k][l][m];
	sort_particle[i][k][l][m]=sort_particle[i][k][l][m]->next_particle;
      }else{
	fprintf(stderr,"Can't allocalte memory\n");
	exit(0);
      }
    }
    
    del_particle_i(particle,i,particle_start);

    if((particle[i]=(Particle*)malloc(sizeof(Particle)))!=NULL){
      particle[i]->next_particle=NULL;
      particle[i]->prev_particle=NULL;
      particle_start[i]=particle[i];
    }else{
      fprintf(stderr,"Can't allocalte memory\n");
      exit(0);
    }

    particle[i]=particle_start[i];

    for(k=0;k<Grid_Nx;k++){
      for(l=0;l<Grid_Ny;l++){
	for(m=0;m<Grid_Nz;m++){
	  if(count[i][k][l][m]!=0){
	    sort_particle[i][k][l][m]=sort_particle_start[i][k][l][m];
	    sort_particle[i][k][l][m]=sort_particle[i][k][l][m]->next_particle;
	    
	    while(sort_particle[i][k][l][m]->next_particle!=NULL){
	      if((particle[i]->next_particle=(Particle*)malloc(sizeof(Particle)))!=NULL){
		memcpy(particle[i]->next_particle,sort_particle[i][k][l][m],sizeof(Particle));
		particle[i]->next_particle->next_particle=NULL;
		particle[i]->next_particle->prev_particle=particle[i];
		particle[i]=particle[i]->next_particle;
	      }else{
		fprintf(stderr,"Can't allocalte memory\n");
		exit(0);
	      }
	      
	      sort_particle[i][k][l][m]=sort_particle[i][k][l][m]->next_particle;
	      free(sort_particle[i][k][l][m]->prev_particle);
	    }
	    
	    if((particle[i]->next_particle=(Particle*)malloc(sizeof(Particle)))!=NULL){
	      memcpy(particle[i]->next_particle,sort_particle[i][k][l][m],sizeof(Particle));
	      particle[i]->next_particle->next_particle=NULL;
	      particle[i]->next_particle->prev_particle=particle[i];
	      particle[i]=particle[i]->next_particle;
	    }else{
	      fprintf(stderr,"Can't allocalte memory\n");
	      exit(0);
	    }
	    
	    free(sort_particle[i][k][l][m]);
	  }
	}
      }
    }

    particle[i]=particle_start[i];
    particle[i]=particle[i]->next_particle;
    free(particle[i]->prev_particle);
    particle[i]->prev_particle=NULL;
    particle_start[i]=particle[i];
  }

  return(0);
}

int all_particles(Particle *ion[],Particle *ion_start[],Particle *electron[],Particle *electron_start[],const int myrank_x,const int myrank_y,int count)
{
  FILE *fp;
  char filename[256];
  int i,k,l;

  for(i=0;i<THREAD_N;i++){
    sprintf(filename,"ap-%d-%d-%d-%d-ion.dat",myrank_x,myrank_y,i,count);
    fp=fopen(filename,"wb");
    
    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){
      fwrite(ion[i],sizeof(Particle),1,fp);
      ion[i]=ion[i]->next_particle;
    }
    fwrite(ion[i],sizeof(Particle),1,fp);
    
    fclose(fp);
  }

  for(i=0;i<THREAD_N;i++){
    sprintf(filename,"ap-%d-%d-%d-%d-electron.dat",myrank_x,myrank_y,i,count);
    fp=fopen(filename,"wb");
    
    electron[i]=electron_start[i];
    while(electron[i]->next_particle!=NULL){
      fwrite(electron[i],sizeof(Particle),1,fp);
      electron[i]=electron[i]->next_particle;
    }
    fwrite(electron[i],sizeof(Particle),1,fp);
    
    fclose(fp);
  }

  return(0);
}

int suspend_job(Particle *ion[],Particle *ion_start[],Particle *electron[],Particle *electron_start[],const int myrank_x,const int myrank_y)
{
  FILE *fp;
  char filename[256];
  int i,k,l;

  sprintf(filename,"suspend-%d-field.dat",myrank_x+myrank_y*PROCESS_Nx);
  fp=fopen(filename,"wb");

  for(k=0;k<Grid_Nx+4;k++){
    for(l=0;l<Grid_Ny+4;l++){
      fwrite(grid[k][l],sizeof(Grid),Grid_Nz+4,fp);
    }
  }

  fclose(fp);

  for(i=0;i<THREAD_N;i++){
    sprintf(filename,"suspend-%d-%d-ion.dat",myrank_x+myrank_y*PROCESS_Nx,i);
    fp=fopen(filename,"wb");
    
    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){
      fwrite(ion[i],sizeof(Particle),1,fp);
      ion[i]=ion[i]->next_particle;
    }
    fwrite(ion[i],sizeof(Particle),1,fp);
    
    fclose(fp);
  }

  for(i=0;i<THREAD_N;i++){
    sprintf(filename,"suspend-%d-%d-electron.dat",myrank_x+myrank_y*PROCESS_Nx,i);
    fp=fopen(filename,"wb");
    
    electron[i]=electron_start[i];
    while(electron[i]->next_particle!=NULL){
      fwrite(electron[i],sizeof(Particle),1,fp);
      electron[i]=electron[i]->next_particle;
    }
    fwrite(electron[i],sizeof(Particle),1,fp);
    
    fclose(fp);
  }

  return(0);
}

int continue_job(Particle *ion[],Particle *ion_start[],Particle *electron[],Particle *electron_start[],const int myrank_x,const int myrank_y)
{
  const gsl_rng_type *T;
  Particle particle;
  int i,k,l,m;
  double x,y,z;
  const int absorb_n=4;
  const double a=1E-6;//1E-9
  FILE *fp;
  char filename[256];

 #pragma omp parallel for private(l,m)
  for(k=0;k<Grid_Nx+4;k++){
    for(l=0;l<Grid_Ny+4;l++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][l][m].bx=0.;
	grid[k][l][m].by=0.;
	grid[k][l][m].bz=0.;
	
	grid[k][l][m].b0x=0.;
	grid[k][l][m].b0y=0.;
	grid[k][l][m].b0z=0.;
	
	grid[k][l][m].ex=0.;
	grid[k][l][m].ey=0.;
	grid[k][l][m].ez=0.;
	
	grid[k][l][m].jix0=0.;
	grid[k][l][m].jiy0=0.;
	grid[k][l][m].jiz0=0.;
	
	grid[k][l][m].jex0=0.;
	grid[k][l][m].jey0=0.;
	grid[k][l][m].jez0=0.;
	
	grid[k][l][m].ni=0.;
	grid[k][l][m].ne=0.;

	grid[k][l][m].phi=0;
      }
    }
  }

#pragma omp parallel for private(l,m,x,y,z)
  for(k=0;k<2*(Grid_Nx+4);k++){
    for(l=0;l<2*(Grid_Ny+4);l++){
      for(m=0;m<2*(Grid_Nz+4);m++){
	sigma[k][l][m]=0.;

	if(k+2*myrank_x*Grid_Nx-4<=2*Absorb_grid){
	  x=(2*Absorb_grid-(k+2*myrank_x*Grid_Nx-4))/4.;
	}else if(k+2*myrank_x*Grid_Nx-4>=2*PROCESS_Nx*Grid_Nx-2*Absorb_grid){
	  x=(k+2*myrank_x*Grid_Nx-4-(2*PROCESS_Nx*Grid_Nx-2*Absorb_grid))/4.;
	}else{
	  x=0.;
	}

	if(l+2*myrank_y*Grid_Ny-4<=2*Absorb_grid){
	  y=(2*Absorb_grid-(l+2*myrank_y*Grid_Ny-4))/4.;
	}else if(l+2*myrank_y*Grid_Ny-4>=2*PROCESS_Ny*Grid_Ny-2*Absorb_grid){
	  y=(l+2*myrank_y*Grid_Ny-4-(2*PROCESS_Ny*Grid_Ny-2*Absorb_grid))/4.;
	}else{
	  y=0.;
	}

	if(m-4<=2*Absorb_grid){
	  z=(2*Absorb_grid-(m-4))/4.;
	}else if(m-4>=2*Grid_Nz-2*Absorb_grid){
	  z=(m-4-(2*Grid_Nz-2*Absorb_grid))/4.;
	}else{
	  z=0.;
	}

	sigma[k][l][m]=a*(0.1*sqrt(pow(x,4)+pow(y,4)+pow(z,4))
			  +0.01*sqrt(pow(x,12)+pow(y,12)+pow(z,12)));
      }
    }
  }

  T=gsl_rng_rand;

  for(i=0;i<THREAD_N;i++){
    rnd_p[i]=gsl_rng_alloc(T);
    rnd_v[i]=gsl_rng_alloc(T);
    rnd_i[i]=gsl_rng_alloc(T);
    gsl_rng_set(rnd_p[i],(myrank_y*PROCESS_Nx+myrank_x)*10000+i*100);
    gsl_rng_set(rnd_v[i],(myrank_y*PROCESS_Nx+myrank_x)*10000+i*100+50);
    gsl_rng_set(rnd_i[i],i);
    particle_flag[i]=1;
  }

  sprintf(filename,"suspend-%d-field.dat",myrank_x+myrank_y*PROCESS_Nx);
  fp=fopen(filename,"rb");

  for(k=0;k<Grid_Nx+4;k++){
    for(l=0;l<Grid_Ny+4;l++){
      fread(grid[k][l],sizeof(Grid),Grid_Nz+4,fp);
    }
  }

  fclose(fp);

  for(i=0;i<THREAD_N;i++){
    ion[i]=ion_start[i]=NULL;
    electron[i]=electron_start[i]=NULL;
  }

  for(i=0;i<THREAD_N;i++){
    if((ion[i]=malloc(sizeof(Particle)))!=NULL){
      ion_start[i]=ion[i];
      ion[i]->prev_particle=NULL;
      ion[i]->next_particle=NULL;
    }
  }

  for(i=0;i<THREAD_N;i++){
    sprintf(filename,"suspend-%d-%d-ion.dat",myrank_x+myrank_y*PROCESS_Nx,i);
    fp=fopen(filename,"rb");
    while(fread(&particle,sizeof(Particle),1,fp)==1){
      if((ion[i]->next_particle=(Particle*)malloc(sizeof(Particle)))!=NULL){
	ion[i]->next_particle->prev_particle=ion[i];

	ion[i]->x=particle.x;
	ion[i]->y=particle.y;
	ion[i]->z=particle.z;
	ion[i]->vx=particle.vx;
	ion[i]->vy=particle.vy;
	ion[i]->vz=particle.vz;
	ion[i]->n=particle.n;
	ion[i]->flag=particle.flag;

	ion[i]=ion[i]->next_particle;
	ion[i]->next_particle=NULL;
      }else{
	exit(0);
      }
    }
    del_particle_i(ion,i,ion_start);

    fclose(fp);
  }

  for(i=0;i<THREAD_N;i++){
    if((electron[i]=malloc(sizeof(Particle)))!=NULL){
      electron_start[i]=electron[i];
      electron[i]->prev_particle=NULL;
      electron[i]->next_particle=NULL;
    }
  }

  for(i=0;i<THREAD_N;i++){
    sprintf(filename,"suspend-%d-%d-electron.dat",myrank_x+myrank_y*PROCESS_Nx,i);
    fp=fopen(filename,"rb");
    while(fread(&particle,sizeof(Particle),1,fp)==1){
      if((electron[i]->next_particle=(Particle*)malloc(sizeof(Particle)))!=NULL){
	electron[i]->next_particle->prev_particle=electron[i];

	electron[i]->x=particle.x;
	electron[i]->y=particle.y;
	electron[i]->z=particle.z;
	electron[i]->vx=particle.vx;
	electron[i]->vy=particle.vy;
	electron[i]->vz=particle.vz;
	electron[i]->n=particle.n;
	electron[i]->flag=particle.flag;

	electron[i]=electron[i]->next_particle;
	electron[i]->next_particle=NULL;
      }else{
	exit(0);
      }
    }
    del_particle_e(electron,i,electron_start);

    fclose(fp);
  }

#pragma omp parallel for private(k,l,m)
  for(i=0;i<THREAD_N;i++){
    for(k=0;k<Grid_Nx;k++){
      for(l=0;l<Grid_Ny;l++){
	for(m=0;m<Grid_Nz;m++){
	  if((sort_particle[i][k][l][m]=(Particle*)malloc(sizeof(Particle)))!=NULL){
	    sort_particle_start[i][k][l][m]=sort_particle[i][k][l][m];
	    sort_particle[i][k][l][m]->prev_particle=NULL;
	    sort_particle[i][k][l][m]->next_particle=NULL;
	  }
	}
      }
    }
  }

  PHI(
      rho_all=malloc(sizeof(double**)*Grid_Nx*PROCESS_Nx);
      
      for(k=0;k<Grid_Nx*PROCESS_Nx;k++){
	rho_all[k]=malloc(sizeof(double*)*Grid_Ny*PROCESS_Ny);
	for(l=0;l<Grid_Ny*PROCESS_Ny;l++){
	  rho_all[k][l]=malloc(sizeof(double)*Grid_Nz);
	  if(rho_all[k][l]==NULL){
	    printf("Don't get MEMORY\n");
	    exit(-1);
	  }
	}
      }
      
      rho_all2=malloc(sizeof(double**)*Grid_Nz);
      
      for(k=0;k<Grid_Nz;k++){
	rho_all2[k]=malloc(sizeof(double*)*Grid_Nx*PROCESS_Nx);
	for(l=0;l<Grid_Nx*PROCESS_Nx;l++){
	  rho_all2[k][l]=malloc(sizeof(double)*Grid_Ny*PROCESS_Ny);
	  if(rho_all2[k][l]==NULL){
	    printf("Don't get MEMORY\n");
	    exit(-1);
	  }
	}
      }
      
      rho_all3=malloc(sizeof(double**)*Grid_Ny*PROCESS_Ny);
      
      for(k=0;k<Grid_Ny*PROCESS_Ny;k++){
	rho_all3[k]=malloc(sizeof(double*)*Grid_Nz);
	for(l=0;l<Grid_Nz;l++){
	  rho_all3[k][l]=malloc(sizeof(double)*Grid_Nx*PROCESS_Nx);
	  if(rho_all3[k][l]==NULL){
	    printf("Don't get MEMORY\n");
	    exit(-1);
	  }
	}
      }
      );

  return(0);
}

gsl_vector *normal_gaussian_vector(const gsl_rng *r,const size_t ndim)
{
  int j;
  gsl_vector *v=gsl_vector_alloc(ndim);

  for(j=0;j<ndim;j++){
    gsl_vector_set(v,j,gsl_ran_gaussian(r,1.));
  }
  return(v);
}

gsl_matrix *covariance_matrix(const size_t ndim,const double *sigma0,const double *rho)
{
  int j,l;
  gsl_matrix *cov=gsl_matrix_alloc(ndim,ndim);

  for(j=0;j<ndim;j++){
    for(l=j;l<ndim;l++){
      if(j==l){
	gsl_matrix_set(cov,j,j,sigma0[j]*sigma0[j]);
      }else{
	gsl_matrix_set(cov,j,l,0./*rho[j+l-1]*sigma0[j]*sigma0[l]*/);
	gsl_matrix_set(cov,l,j,gsl_matrix_get(cov,j,l));
      }
    }
  }

  return(cov);
}

void cholesky_decomp_ltri(gsl_matrix *cov)
{
  int j,l;

  gsl_linalg_cholesky_decomp(cov);

  for(j=0;j<cov->size1;j++){
    for(l=j+1;l<cov->size2;l++){
      gsl_matrix_set(cov,j,l,0.);
    }
  }
}

double *ran_multivar_gaussian(const gsl_rng *rng,const size_t ndim,const double *mu,const double *sigma0,const double *rho)
{
  int j;

  gsl_matrix *cov;
  gsl_vector *normal;
  gsl_matrix *chol;
  double *x;
  gsl_vector_view vx;
  
  normal=normal_gaussian_vector(rng,ndim);
  
  cov=covariance_matrix(ndim,sigma0,rho);

  cholesky_decomp_ltri(cov);
  chol=cov;
  
  x=(double *)malloc(ndim*sizeof(double));
  vx=gsl_vector_view_array(x,ndim);
  
  gsl_blas_dgemv(CblasNoTrans,1.,chol,normal,0.,&vx.vector);
  gsl_vector_free(normal);
  gsl_matrix_free(cov);
  
  for(j=0;j<ndim;j++){
    x[j]+=mu[j];
  }
  
  return(x);
}

/*******************************************************************
 fft for z-axis
 ******************************************************************/
int fft_z(double rho[][Grid_Ny][Grid_Nz])
{
  int k,l,m;
  int local_thread;
  double data[THREAD_N][Grid_Nz+1];
  
#pragma omp parallel for private(l,m,local_thread)
  for(k=0;k<Grid_Nx;k++){
    for(l=0;l<Grid_Ny;l++){
#ifdef _OPENMP
      local_thread=omp_get_thread_num();
#else
      local_thread=0;
#endif
      for(m=0;m<Grid_Nz;m++){
	data[local_thread][m+1]=rho[k][l][m];
      }
      
      sinft(Grid_Nz,data[local_thread]);
      
      for(m=0;m<Grid_Nz;m++){
	rho[k][l][m]=data[local_thread][m+1];
      }
    }
  }
  
  return(0);
}

/*******************************************************************
 fft for y-axis
******************************************************************/
int fft_y(double rho2[][Grid_Nx*PROCESS_Nx/PROCESS_Ny][Grid_Ny*PROCESS_Ny])
{
  int k,l,m;
  int local_thread;
  double data[THREAD_N][Grid_Ny*PROCESS_Ny+1];
  
#pragma omp parallel for private(l,m,local_thread)
  for(k=0;k<Grid_Nz/PROCESS_Nx;k++){
    for(l=0;l<Grid_Nx*PROCESS_Nx/PROCESS_Ny;l++){
#ifdef _OPENMP
      local_thread=omp_get_thread_num();
#else
      local_thread=0;
#endif
      for(m=0;m<Grid_Ny*PROCESS_Ny;m++){
	data[local_thread][m+1]=rho2[k][l][m];
      }
      
      sinft(Grid_Ny*PROCESS_Ny,data[local_thread]);
    
      for(m=0;m<Grid_Ny*PROCESS_Ny;m++){
	rho2[k][l][m]=data[local_thread][m+1];
      }
    }
  }
  
  return(0);
}

/*******************************************************************
 fft for x-axis
******************************************************************/
int fft_x(double rho3[][Grid_Nz/PROCESS_Ny][Grid_Nx*PROCESS_Nx])
{
  int k,l,m;
  int local_thread;
  double data[THREAD_N][Grid_Nx*PROCESS_Nx+1];
  
#pragma omp parallel for private(l,m,local_thread)
  for(k=0;k<Grid_Ny*PROCESS_Ny/PROCESS_Nx;k++){
    for(l=0;l<Grid_Nz/PROCESS_Ny;l++){
#ifdef _OPENMP
      local_thread=omp_get_thread_num();
#else
      local_thread=0;
#endif
      for(m=0;m<Grid_Nx*PROCESS_Nx;m++){
	data[local_thread][m+1]=rho3[k][l][m];
      }
      
      sinft(Grid_Nx*PROCESS_Nx,data[local_thread]);
    
      for(m=0;m<Grid_Nx*PROCESS_Nx;m++){
	rho3[k][l][m]=data[local_thread][m+1];
      }
    }
  }
  
  return(0);
}

/*******************************************************************
 inverse fft for z-axis
******************************************************************/
int inv_fft_z(double rho[][Grid_Ny][Grid_Nz])
{
  int k,l,m;
  int local_thread;
  double data[THREAD_N][Grid_Nz+1];
  
#pragma omp parallel for private(l,m,local_thread)
  for(k=0;k<Grid_Nx;k++){
    for(l=0;l<Grid_Ny;l++){
#ifdef _OPENMP
      local_thread=omp_get_thread_num();
#else
      local_thread=0;
#endif
      for(m=0;m<Grid_Nz;m++){
	data[local_thread][m+1]=rho[k][l][m];
      }
      
      sinft(Grid_Nz,data[local_thread]);
      
      for(m=0;m<Grid_Nz;m++){
	rho[k][l][m]=2.*data[local_thread][m+1]/Grid_Nz;
      }
    }
  }
  
  return(0);
}

/*******************************************************************
 inverce fft for y-axis
******************************************************************/
int inv_fft_y(double rho2[][Grid_Nx*PROCESS_Nx/PROCESS_Ny][Grid_Ny*PROCESS_Ny])
{
  int k,l,m;
  int local_thread;
  double data[THREAD_N][Grid_Ny*PROCESS_Ny+1];
  
#pragma omp parallel for private(l,m,local_thread)
  for(k=0;k<Grid_Nz/PROCESS_Nx;k++){
    for(l=0;l<Grid_Nx*PROCESS_Nx/PROCESS_Ny;l++){
#ifdef _OPENMP
      local_thread=omp_get_thread_num();
#else
      local_thread=0;
#endif
      for(m=0;m<Grid_Ny*PROCESS_Ny;m++){
	data[local_thread][m+1]=rho2[k][l][m];
      }
      
      sinft(Grid_Ny*PROCESS_Ny,data[local_thread]);
    
      for(m=0;m<Grid_Ny*PROCESS_Ny;m++){
	rho2[k][l][m]=2.*data[local_thread][m+1]/(Grid_Ny*PROCESS_Ny);
      }
    }
  }
  
  return(0);
}

/*******************************************************************
 inverce fft for x-axis
******************************************************************/
int inv_fft_x(double rho3[][Grid_Nz/PROCESS_Ny][Grid_Nx*PROCESS_Nx])
{
  int k,l,m;
  int local_thread;
  double data[THREAD_N][Grid_Nx*PROCESS_Nx+1];
  
#pragma omp parallel for private(l,m,local_thread)
  for(k=0;k<Grid_Ny*PROCESS_Ny/PROCESS_Nx;k++){
    for(l=0;l<Grid_Nz/PROCESS_Ny;l++){
#ifdef _OPENMP
      local_thread=omp_get_thread_num();
#else
      local_thread=0;
#endif
      for(m=0;m<Grid_Nx*PROCESS_Nx;m++){
	data[local_thread][m+1]=rho3[k][l][m];
      }
      
      sinft(Grid_Nx*PROCESS_Nx,data[local_thread]);
    
      for(m=0;m<Grid_Nx*PROCESS_Nx;m++){
	rho3[k][l][m]=2.*data[local_thread][m+1]/(Grid_Nx*PROCESS_Nx);
      }
    }
  }
  
  return(0);
}

/*******************************************************************
fast sin fouir transform
******************************************************************/
int sinft(int n,double y[])
{
  int j,n2=n+2;
  double sum,y1,y2;
  double theta,wi=0.,wr=1.,wpi,wpr,wtemp;

  theta=Pi/(double)n;
  wtemp=sin(0.5*theta);
  wpr=-2.0*wtemp*wtemp;
  wpi=sin(theta);
  y[1]=0.0;

  for(j=2;j<=(n>>1)+1;j++){
    wr=(wtemp=wr)*wpr-wi*wpi+wr;
    wi=wi*wpr+wtemp*wpi+wi;
    y1=wi*(y[j]+y[n2-j]);
    y2=0.5*(y[j]-y[n2-j]);
    y[j]=y1+y2;
    y[n2-j]=y1-y2;
  }
  realft(y,n,1);
  y[1]*=0.5;
  sum=y[2]=0.0;

  for(j=1;j<=n-1;j+=2){
    sum+=y[j];
    y[j]=y[j+1];
    y[j+1]=sum;
  }
  return(0);
}

/*******************************************************************
fft for real number
******************************************************************/
int realft(double data[],int n,int isign)
{
  int i,i1,i2,i3,i4,np3;
  double c1=0.5,c2,h1r,h1i,h2r,h2i;
  double wr,wi,wpr,wpi,wtemp,theta;

  theta=Pi/(double)(n>>1);

  if(isign==1){
    c2=-0.5;
    four1(data,n>>1,1);
  }else{
    c2=0.5;
    theta=-theta;
  }

  wtemp=sin(0.5*theta);
  wpr=-2.0*wtemp*wtemp;
  wpi=sin(theta);
  wr=1.+wpr;
  wi=wpi;
  np3=n+3;

  for(i=2;i<=(n>>2);i++){
    i4=1+(i3=np3-(i2=1+(i1=i+i-1)));
    h1r=c1*(data[i1]+data[i3]);
    h1i=c1*(data[i2]-data[i4]);
    h2r=-c2*(data[i2]+data[i4]);
    h2i=c2*(data[i1]-data[i3]);
    data[i1]=h1r+wr*h2r-wi*h2i;
    data[i2]=h1i+wr*h2i+wi*h2r;
    data[i3]=h1r-wr*h2r+wi*h2i;
    data[i4]=-h1i+wr*h2i+wi*h2r;
    wr=(wtemp=wr)*wpr-wi*wpi+wr;
    wi=wi*wpr+wtemp*wpi+wi;
  }

  if(isign==1){
    data[1]=(h1r=data[1])+data[2];
    data[2]=h1r-data[2];
  }else{
    data[1]=c1*((h1r=data[1])+data[2]);
    data[2]=c1*(h1r-data[2]);
    four1(data,n>>1,-1);
  }
  return(0);
}

/*******************************************************************
fft
******************************************************************/
int four1(double data[],int nn,int isign)
{
  int n,mmax,m,j,istep,i;
  double wtemp,wr,wpr,wpi,wi,theta;
  double tempr,tempi;

  n=nn<<1;
  j=1;
  for(i=1;i<n;i+=2){
    if(j>i){
      SWAP(data[j],data[i]);
      SWAP(data[j+1],data[i+1]);
    }
    m=n>>1;
    while(m>=2&&j>m){
      j-=m;
      m>>=1;
    }
    j+=m;
  }
  mmax=2;
  while(n>mmax){
    istep=mmax<<1;
    theta=isign*2.*Pi/mmax;
    wtemp=sin(0.5*theta);
    wpr=-2.0*wtemp*wtemp;
    wpi=sin(theta);
    wr=1.;
    wi=0.;
    for(m=1;m<mmax;m+=2){
      for(i=m;i<=n;i+=istep){
	j=i+mmax;
	tempr=wr*data[j]-wi*data[j+1];
	tempi=wr*data[j+1]+wi*data[j];
	data[j]=data[i]-tempr;
	data[j+1]=data[i+1]-tempi;
	data[i]+=tempr;
	data[i+1]+=tempi;
      }
      wr=(wtemp=wr)*wpr-wi*wpi+wr;
      wi=wi*wpr+wtemp*wpi+wi;
    }
    mmax=istep;
  }
  return(0);
}

int renew_phi3(const int myrank_x,const int myrank_y)
{
  int i,j,k,l,m;
  double A;
  int src,dest,tag,count;
  MPI_Status stat[PROCESS_Nx*PROCESS_Ny];
  MPI_Request request[PROCESS_Nx*PROCESS_Ny];
  MPI_Status stat2[PROCESS_Nx*PROCESS_Ny];
  MPI_Request request2[PROCESS_Nx*PROCESS_Ny];
  double rho[Grid_Nx][Grid_Ny][Grid_Nz];
  double rho2[Grid_Nz/PROCESS_Nx][Grid_Nx*PROCESS_Nx/PROCESS_Ny][Grid_Ny*PROCESS_Ny];
  double rho3[Grid_Ny*PROCESS_Ny/PROCESS_Nx][Grid_Nz/PROCESS_Ny][Grid_Nx*PROCESS_Nx];
  
  tag=1000;
  
#pragma omp parallel for private(l,m)
  for(k=0;k<Grid_Nx;k++){
    for(l=0;l<Grid_Ny;l++){
      for(m=0;m<Grid_Nz;m++){
	rho[k][l][m]=0.;
      }
    }
  }

#pragma omp parallel for private(l,m)
  for(k=2;k<Grid_Nx+2;k++){
    for(l=2;l<Grid_Ny+2;l++){
      for(m=Absorb_grid3+2;m<Grid_Nz+2-Absorb_grid3;m++){
	if(k+myrank_x*Grid_Nx-2>Absorb_grid3&&k+myrank_x*Grid_Nx-2<Grid_Nx*PROCESS_Nx-Absorb_grid3&&
	   l+myrank_y*Grid_Ny-2>Absorb_grid3&&l+myrank_y*Grid_Ny-2<Grid_Ny*PROCESS_Ny-Absorb_grid3){
	  rho[k-2][l-2][m-2]=-(-q*(grid[k][l][m].ni-grid[k][l][m].ne)/e0
			       +(-grid[k+2][l][m].ex+27*grid[k+1][l][m].ex-27*grid[k][l][m].ex+grid[k-1][l][m].ex
				 -grid[k][l+2][m].ey+27*grid[k][l+1][m].ey-27*grid[k][l][m].ey+grid[k][l-1][m].ey
				 -grid[k][l][m+2].ez+27*grid[k][l][m+1].ez-27*grid[k][l][m].ez+grid[k][l][m-1].ez)/(24.*dx));

	  /* +(grid[k+1][l][m].ex-grid[k][l][m].ex+
			    grid[k][l+1][m].ey-grid[k][l][m].ey+
			    grid[k][l][m+1].ez-grid[k][l][m].ez)/dx);*/
	}
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  fft_z(rho);

  if(myrank_x==0&&myrank_y==0){
    for(k=0;k<Grid_Nx;k++){
      for(l=0;l<Grid_Ny;l++){
	for(m=0;m<Grid_Nz;m++){
	  rho_all[k][l][m]=rho[k][l][m];
	}
      }
    }
    
    for(i=0;i<PROCESS_Nx;i++){
      for(j=0;j<PROCESS_Ny;j++){
	if(i!=0||j!=0){
	  MPI_Irecv(&rho,Grid_Nx*Grid_Ny*Grid_Nz,MPI_DOUBLE,i+j*PROCESS_Nx,tag,MPI_COMM_WORLD,&request[i+j*PROCESS_Nx]);
	  MPI_Wait(&request[i+j*PROCESS_Nx],&stat[i+j*PROCESS_Nx]);
	  
	  for(k=0;k<Grid_Nx;k++){
	    for(l=0;l<Grid_Ny;l++){
	      for(m=0;m<Grid_Nz;m++){
		rho_all[i*Grid_Nx+k][j*Grid_Ny+l][m]=rho[k][l][m];
	      }
	    }
	  }
	}
      }
    }

    for(k=0;k<Grid_Nx*PROCESS_Nx;k++){
      for(l=0;l<Grid_Ny*PROCESS_Ny;l++){
	for(m=0;m<Grid_Nz;m++){
	  rho_all2[m][k][l]=rho_all[k][l][m];
	}
      }
    }

    for(i=0;i<PROCESS_Nx;i++){
      for(j=0;j<PROCESS_Ny;j++){
	if(i!=0||j!=0){
	  for(k=0;k<Grid_Nz/PROCESS_Nx;k++){
	    for(l=0;l<Grid_Nx*PROCESS_Nx/PROCESS_Ny;l++){
	      for(m=0;m<Grid_Ny*PROCESS_Ny;m++){
		rho2[k][l][m]=rho_all2[i*Grid_Nz/PROCESS_Nx+k][j*Grid_Nx*PROCESS_Nx/PROCESS_Ny+l][m];
	      }
	    }
	  }
	  MPI_Send(&rho2,Grid_Nx*Grid_Ny*Grid_Nz,MPI_DOUBLE,i+j*PROCESS_Nx,tag,MPI_COMM_WORLD);
	}
      }
    }

    for(k=0;k<Grid_Nz/PROCESS_Nx;k++){
      for(l=0;l<Grid_Nx*PROCESS_Nx/PROCESS_Ny;l++){
	for(m=0;m<Grid_Ny*PROCESS_Ny;m++){
	  rho2[k][l][m]=rho_all2[k][l][m];
	}
      }
    }
  }else{
    MPI_Send(&rho,Grid_Nx*Grid_Ny*Grid_Nz,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);

    MPI_Irecv(&rho2,Grid_Nx*Grid_Ny*Grid_Nz,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&request[0]);

    MPI_Wait(&request[0],&stat[0]);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  
  fft_y(rho2);

  if(myrank_x==0&&myrank_y==0){
    for(k=0;k<Grid_Nz/PROCESS_Nx;k++){
      for(l=0;l<Grid_Nx*PROCESS_Nx/PROCESS_Ny;l++){
	for(m=0;m<Grid_Ny*PROCESS_Ny;m++){
	  rho_all2[k][l][m]=rho2[k][l][m];
	}
      }
    }
    
    for(i=0;i<PROCESS_Nx;i++){
      for(j=0;j<PROCESS_Ny;j++){
	if(i!=0||j!=0){
	  MPI_Irecv(&rho2,Grid_Nx*Grid_Ny*Grid_Nz,MPI_DOUBLE,i+j*PROCESS_Nx,tag,MPI_COMM_WORLD,&request[i+j*PROCESS_Nx]);
	  MPI_Wait(&request[i+j*PROCESS_Nx],&stat[i+j*PROCESS_Nx]);
	  
	  for(k=0;k<Grid_Nz/PROCESS_Nx;k++){
	    for(l=0;l<Grid_Nx*PROCESS_Nx/PROCESS_Ny;l++){
	      for(m=0;m<Grid_Ny*PROCESS_Ny;m++){
		rho_all2[i*Grid_Nz/PROCESS_Nx+k][j*Grid_Nx*PROCESS_Nx/PROCESS_Ny+l][m]=rho2[k][l][m];
	      }
	    }
	  }
	}
      }
    }

    for(k=0;k<Grid_Nz;k++){
      for(l=0;l<Grid_Nx*PROCESS_Nx;l++){
	for(m=0;m<Grid_Ny*PROCESS_Ny;m++){
	  rho_all3[m][k][l]=rho_all2[k][l][m];
	}
      }
    }

    for(i=0;i<PROCESS_Nx;i++){
      for(j=0;j<PROCESS_Ny;j++){
	if(i!=0||j!=0){
	  for(k=0;k<Grid_Ny*PROCESS_Ny/PROCESS_Nx;k++){
	    for(l=0;l<Grid_Nz/PROCESS_Ny;l++){
	      for(m=0;m<Grid_Nx*PROCESS_Nx;m++){
		rho3[k][l][m]=rho_all3[i*Grid_Ny*PROCESS_Ny/PROCESS_Nx+k][j*Grid_Nz/PROCESS_Ny+l][m];
	      }
	    }
	  }
	  MPI_Send(&rho3,Grid_Nx*Grid_Ny*Grid_Nz,MPI_DOUBLE,i+j*PROCESS_Nx,tag,MPI_COMM_WORLD);
	}
      }
    }

    for(k=0;k<Grid_Ny*PROCESS_Ny/PROCESS_Nx;k++){
      for(l=0;l<Grid_Nz/PROCESS_Ny;l++){
	for(m=0;m<Grid_Nx*PROCESS_Nx;m++){
	  rho3[k][l][m]=rho_all3[k][l][m];
	}
      }
    }
  }else{
    MPI_Send(&rho2,Grid_Nx*Grid_Ny*Grid_Nz,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);

    MPI_Irecv(&rho3,Grid_Nx*Grid_Ny*Grid_Nz,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&request[0]);

    MPI_Wait(&request[0],&stat[0]);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  
  fft_x(rho3);

#pragma omp parallel for private(l,m,A)
  for(k=0;k<Grid_Ny*PROCESS_Ny/PROCESS_Nx;k++){
    for(l=0;l<Grid_Nz/PROCESS_Ny;l++){
      for(m=0;m<Grid_Nx*PROCESS_Nx;m++){
	A=-(pow(sin(Pi*(double)(k+myrank_x*Grid_Ny*PROCESS_Ny/PROCESS_Nx)/(double)(Grid_Ny*PROCESS_Ny)/2.),2)
	    +pow(sin(Pi*(double)(l+myrank_y*Grid_Nz/PROCESS_Ny)/(double)(Grid_Nz)/2.),2)
	    +pow(sin(Pi*(double)m/(double)(Grid_Nx*PROCESS_Nx)/2.),2))/pow(dx/2.,2);
	
	if(A==0.){
	  rho3[k][l][m]=0.;
	}else{
	  rho3[k][l][m]=rho3[k][l][m]/A;       
	} 
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  
  inv_fft_x(rho3);

  if(myrank_x==0&&myrank_y==0){
    for(k=0;k<Grid_Ny*PROCESS_Ny/PROCESS_Nx;k++){
      for(l=0;l<Grid_Nz/PROCESS_Ny;l++){
	for(m=0;m<Grid_Nx*PROCESS_Nx;m++){
	  rho_all3[k][l][m]=rho3[k][l][m];
	}
      }
    }
    
    for(i=0;i<PROCESS_Nx;i++){
      for(j=0;j<PROCESS_Ny;j++){
	if(i!=0||j!=0){
	  MPI_Irecv(&rho3,Grid_Nx*Grid_Ny*Grid_Nz,MPI_DOUBLE,i+j*PROCESS_Nx,tag,MPI_COMM_WORLD,&request[i+j*PROCESS_Nx]);
	  MPI_Wait(&request[i+j*PROCESS_Nx],&stat[i+j*PROCESS_Nx]);
	  
	  for(k=0;k<Grid_Ny*PROCESS_Ny/PROCESS_Nx;k++){
	    for(l=0;l<Grid_Nz/PROCESS_Ny;l++){
	      for(m=0;m<Grid_Nx*PROCESS_Nx;m++){
		rho_all3[i*Grid_Ny*PROCESS_Ny/PROCESS_Nx+k][j*Grid_Nz/PROCESS_Ny+l][m]=rho3[k][l][m];
	      }
	    }
	  }
	}
      }
    }
    
    for(k=0;k<Grid_Ny*PROCESS_Ny;k++){
      for(l=0;l<Grid_Nz;l++){
	for(m=0;m<Grid_Nx*PROCESS_Nx;m++){
	  rho_all2[l][m][k]=rho_all3[k][l][m];
	}
      }
    }
    
    for(i=0;i<PROCESS_Nx;i++){
      for(j=0;j<PROCESS_Ny;j++){
	if(i!=0||j!=0){
	  for(k=0;k<Grid_Nz/PROCESS_Nx;k++){
	    for(l=0;l<Grid_Nx*PROCESS_Nx/PROCESS_Ny;l++){
	      for(m=0;m<Grid_Ny*PROCESS_Ny;m++){
		rho2[k][l][m]=rho_all2[i*Grid_Nz/PROCESS_Nx+k][j*Grid_Nx*PROCESS_Nx/PROCESS_Ny+l][m];
	      }
	    }
	  }
	  MPI_Send(&rho2,Grid_Nx*Grid_Ny*Grid_Nz,MPI_DOUBLE,i+j*PROCESS_Nx,tag,MPI_COMM_WORLD);
	}
      }
    }

    for(k=0;k<Grid_Nz/PROCESS_Nx;k++){
      for(l=0;l<Grid_Nx*PROCESS_Nx/PROCESS_Ny;l++){
	for(m=0;m<Grid_Ny*PROCESS_Ny;m++){
	  rho2[k][l][m]=rho_all2[k][l][m];
	}
      }
    }
  }else{
    MPI_Send(&rho3,Grid_Nx*Grid_Ny*Grid_Nz,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);

    MPI_Irecv(&rho2,Grid_Nx*Grid_Ny*Grid_Nz,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&request[0]);

    MPI_Wait(&request[0],&stat[0]);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  inv_fft_y(rho2);
  
  if(myrank_x==0&&myrank_y==0){
    for(k=0;k<Grid_Nz/PROCESS_Nx;k++){
      for(l=0;l<Grid_Nx*PROCESS_Nx/PROCESS_Ny;l++){
	for(m=0;m<Grid_Ny*PROCESS_Ny;m++){
	  rho_all2[k][l][m]=rho2[k][l][m];
	}
      }
    }
    
    for(i=0;i<PROCESS_Nx;i++){
      for(j=0;j<PROCESS_Ny;j++){
	if(i!=0||j!=0){
	  MPI_Irecv(&rho2,Grid_Nx*Grid_Ny*Grid_Nz,MPI_DOUBLE,i+j*PROCESS_Nx,tag,MPI_COMM_WORLD,&request[i+j*PROCESS_Nx]);
	  MPI_Wait(&request[i+j*PROCESS_Nx],&stat[i+j*PROCESS_Nx]);
	  
	  for(k=0;k<Grid_Nz/PROCESS_Nx;k++){
	    for(l=0;l<Grid_Nx*PROCESS_Nx/PROCESS_Ny;l++){
	      for(m=0;m<Grid_Ny*PROCESS_Ny;m++){
		rho_all2[i*Grid_Nz/PROCESS_Nx+k][j*Grid_Nx*PROCESS_Nx/PROCESS_Ny+l][m]=rho2[k][l][m];
	      }
	    }
	  }
	}
      }
    }

    for(k=0;k<Grid_Nz;k++){
      for(l=0;l<Grid_Nx*PROCESS_Nx;l++){
	for(m=0;m<Grid_Ny*PROCESS_Ny;m++){
	  rho_all[l][m][k]=rho_all2[k][l][m];
	}
      }
    }

    for(i=0;i<PROCESS_Nx;i++){
      for(j=0;j<PROCESS_Ny;j++){
	if(i!=0||j!=0){
	  for(k=0;k<Grid_Nx;k++){
	    for(l=0;l<Grid_Ny;l++){
	      for(m=0;m<Grid_Nz;m++){
		rho[k][l][m]=rho_all[i*Grid_Nx+k][j*Grid_Ny+l][m];
	      }
	    }
	  }
	  MPI_Send(&rho,Grid_Nx*Grid_Ny*Grid_Nz,MPI_DOUBLE,i+j*PROCESS_Nx,tag,MPI_COMM_WORLD);
	}
      }
    }

    for(k=0;k<Grid_Nx;k++){
      for(l=0;l<Grid_Ny;l++){
	for(m=0;m<Grid_Nz;m++){
	  rho[k][l][m]=rho_all[k][l][m];
	}
      }
    }
  }else{
    MPI_Send(&rho2,Grid_Nx*Grid_Ny*Grid_Nz,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);

    MPI_Irecv(&rho,Grid_Nx*Grid_Ny*Grid_Nz,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&request[0]);

    MPI_Wait(&request[0],&stat[0]);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  inv_fft_z(rho);
  
#pragma omp parallel for private(l,m)
  for(k=0;k<Grid_Nx;k++){
    for(l=0;l<Grid_Ny;l++){
      for(m=0;m<Grid_Nz;m++){
	grid[k+2][l+2][m+2].phi=rho[k][l][m];
      }
    }
  }

  return(0);
}
