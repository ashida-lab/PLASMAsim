#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <signal.h>

/*******************************************************************
GSL library  gcc -lgsl -lgslcblas
******************************************************************/
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_const_mksa.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/*******************************************************************
openMP library  gcc -fopenmp or icc -openmp

mpicc 2d_pic278_mpi.c -o 2d_pic278_mpi -lm -lgsl -lgslcblas -O3 -openmp -no-prec-div -no-prec-sqrt -xSSE4.2 -axSSE4.2 -static-intel -ipo -opt-mem-bandwidth2  -opt-calloc -unroll-aggressive -ltcmalloc
******************************************************************/
#ifdef _OPENMP
#include <omp.h>
#endif

#include <mpi.h>

#define version 1278

#define q GSL_CONST_MKSA_ELECTRON_CHARGE
#define mi GSL_CONST_MKSA_MASS_PROTON
#define me GSL_CONST_MKSA_MASS_ELECTRON //(GSL_CONST_MKSA_MASS_PROTON/100.)//GSL_CONST_MKSA_MASS_ELECTRON  /**/
#define mu0 GSL_CONST_MKSA_VACUUM_PERMEABILITY
#define e0 GSL_CONST_MKSA_VACUUM_PERMITTIVITY
#define kb GSL_CONST_MKSA_BOLTZMANN
#define Pi M_PI
#define C 3.0E8

//Parameters for two particle or boltzman
#define R 75.0//2.//coil radius [m]
#define I 4.0E3 //coil current x turn [A T]     
#define alpha 0. //////////// pi/2=atack 0,0=atack pi/2

#define dx 30. //60.//grid width [m]  if two particle or boltzman dx < debye length
#define dtf 0.6E-7
#define dti (0.6E-7)//0.6E-7
#define dte (0.6E-7)
#define Grid_Nx 16 //grid number x,y-axis
#define Grid_Nz 512 //grid number z-axis


#define Np 4
#define N Np*Grid_Nx //ion particle number 
#define Step_p Np*Grid_Nz
#define V 5.0E5  //sun wind velocity  z-axis
#define Nall 5.0E6 //sun wind particle density
#define Ti 1.0E6
#define Te 1.0E6


#define N0i Nall/Np/Np
#define N0e Nall/Np/Np


#define IMF_x 0.//3.5E-9
#define IMF_y 0.
#define IMF_z 0.//3.5E-9

#define PACK_N 40000

#define first_filter 0 //strength of first degital filter
#define latter_filter 0 //strength of latter degital filter

#define Step 10000001

#define eps 1.0E-10

#define Absorb_grid 32  //field
#define Absorb_grid2 16  //particle
#define Absorb_grid3 32 //phi

#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr
#define ABSORB(x) x

#define THREAD_N 2
#define PROCESS_N 32


/*******************************************************************
type holding particle infomation
******************************************************************/
typedef struct particle_info{
  double x,z;
  double vx,vy,vz;
  double n;
  int flag;
  struct particle_info *prev_particle;
  struct particle_info *next_particle;
}Particle;

typedef struct{
  double x,z;
  double vx,vy,vz;
  double n;
}PParticle;
 
/*******************************************************************
type holding grid infomation
******************************************************************/
typedef struct{
  double b0x,b0y,b0z;//initial magnetic field
  double bx,by,bz;//total magnetic field   b0-b=induced magnetic field
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

double init_L();

int main(int argc,char *argv[])
{
  int myid,myrank,p;
  double b0;
  char filename[256];
  FILE *fp1;

  MPI_Init(&argc,&argv);

  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&p);

  myrank=myid;

  if(p!=PROCESS_N){
    fprintf(stderr,"Process number is falut\n");
    exit(0);
  }

  if(myrank==0){
    if(1.414*C*dtf>dx){
      printf("Courant error\n");
      //exit(-1);
    }else{
      printf("dx/dtf/C/1.414 = %f >1\n",dx/dtf/C/1.414);
    }

    fprintf(stderr,"%d L=%E Debye=%f\n",version,init_L(),sqrt(e0*kb*Te/Nall/q/q));
    fprintf(stderr,"B@mp=%E\n",b0=sqrt(2*mu0*Nall*mi*V*V));
    fprintf(stderr,"wep=%E wec=%E\n",sqrt(Nall*q*q/e0/me),q/me*b0);
    fprintf(stderr,"wip=%E wic=%E\n",sqrt(Nall*q*q/e0/mi),q/mi*b0);
    fprintf(stderr,"tec=%E tic=%E\n",2.*Pi*me/q/b0,2.*Pi*mi/q/b0);
    fprintf(stderr,"tep=%E tip=%E\n",2.*Pi/sqrt(Nall*q*q/e0/me),2.*Pi/sqrt(Nall*q*q/e0/mi));
    fprintf(stderr,"vi_th=%E ve_th=%E\n",sqrt(2.*kb*Ti/mi),sqrt(2.*kb*Te/me));
    fprintf(stderr,"vs=%E\n",sqrt(2.*kb*Ti/mi+kb*Te/mi));
    fprintf(stderr,"Required steps %d\n",(int)(dx*Grid_Nz/V/dtf));

    sprintf(filename,"pic%d-f.txt",version);
    fp1=fopen(filename,"w");
    fclose(fp1);

    sprintf(filename,"pic%d-f2.txt",version);
    fp1=fopen(filename,"w");
    fclose(fp1);

    sprintf(filename,"pic%d-f3.txt",version);
    fp1=fopen(filename,"w");
    fclose(fp1);
  }

#ifdef _OPENMP
  omp_set_num_threads(THREAD_N);
#endif

  main_func(myrank,p);

  MPI_Finalize();

  return(0);
}

double init_L()
{
  double k;

  //k=pow(mu0*pow(M,2)/(8*Pi*Pi*Nall*mi*V*V),1./6.);

  k=pow(mu0*4.*I*I*R*R/(Pi*Pi*Nall*mi*V*V),1./4.);
  //k=pow(mu0*4.*I*I*R*R/(2.*Pi*Pi*Nall*mi*V*V),1./4.);
  return(k);
}

int main_func(const int myrank,const int p)
{
  int i;
  Grid grid[Grid_Nx+4][Grid_Nz+4];
  double sigma[2*(Grid_Nx+4)][2*(Grid_Nz+4)];
  Particle *ion[THREAD_N];
  Particle *ion_start[THREAD_N];
  Particle *electron[THREAD_N];
  Particle *electron_start[THREAD_N];
  gsl_rng *rnd_p[THREAD_N];
  gsl_rng *rnd_v[THREAD_N];

  printf("my rank is %d/%d\n",myrank,p);

  init_grid(grid,sigma,myrank,p);

  init_particle(ion,ion_start,electron,electron_start,rnd_p,rnd_v,myrank);
  
  fdtd(grid,sigma,myrank,p,ion,ion_start,electron,electron_start,rnd_p,rnd_v);

  return(0);
}

int init_grid(Grid grid[][Grid_Nz+4],double sigma[][2*(Grid_Nz+4)],const int myrank,const int p)
{
  int k,m;
  const int absorb_n=4;
  const double a=1E-9;
  double x,z;
  double bx,bz;

  for(k=0;k<Grid_Nx+4;k++){
    for(m=0;m<Grid_Nz+4;m++){
      grid[k][m].bx=0.;
      grid[k][m].by=IMF_y;
      grid[k][m].bz=0.;

      grid[k][m].b0x=0.;
      grid[k][m].b0y=IMF_y;
      grid[k][m].b0z=0.;

      grid[k][m].ex=0.;
      grid[k][m].ey=0.;
      grid[k][m].ez=0.;
      
      grid[k][m].jix0=0.;
      grid[k][m].jiy0=0.;
      grid[k][m].jiz0=0.;
      
      grid[k][m].jex0=0.;
      grid[k][m].jey0=0.;
      grid[k][m].jez0=0.;
      
      grid[k][m].ni=0.;
      grid[k][m].ne=0.;

      grid[k][m].phi=0.;
    }
  }

#pragma omp parallel for private(m,x,z)
  for(k=0;k<2*(Grid_Nx+4);k++){
    for(m=0;m<2*(Grid_Nz+4);m++){
      sigma[k][m]=0.;
      
      if(k+2*myrank*Grid_Nx-4<=2*Absorb_grid){
	x=(2*Absorb_grid-(k+2*myrank*Grid_Nx-4))/4.;
      }else if(k+2*myrank*Grid_Nx-4>=2*PROCESS_N*Grid_Nx-2*Absorb_grid){
	x=((k+2*myrank*Grid_Nx-4)-(2*PROCESS_N*Grid_Nx-2*Absorb_grid))/4.;
      }else{
	x=0.;
      }
     
      
      if(m-4<=2*Absorb_grid){
	z=(2*Absorb_grid-(m-4))/4.;
      }else if(m-4>=2*Grid_Nz-2*Absorb_grid){
	z=((m-4)-(2*Grid_Nz-2*Absorb_grid))/4.;
      }else{
	z=0.;
      }
      
      sigma[k][m]=a*(0.1*sqrt(pow(x,4)+pow(z,4))+0.01*sqrt(pow(x,12)+pow(z,12)));

    }
  }

  /*if(myrank==0){
    for(k=0;k<2*(Grid_Nx+4);k++){
      for(m=0;m<2*(Grid_Nz+4);m++){
	sigma[k][m]=0.;
	if(k<=2*Absorb_grid&&m<=2*Absorb_grid){
	  sigma[k][m]=a*(0.1*sqrt(pow((2*Absorb_grid-k)/4.,4)+pow((2*Absorb_grid-m)/4.,4))
			 +0.01*sqrt(pow((2*Absorb_grid-k)/4.,12)+pow((2*Absorb_grid-m)/4.,12)));
	}else if(k<=2*Absorb_grid&&m>=2*(Grid_Nz+4)-2*Absorb_grid){
	  sigma[k][m]=a*(0.1*sqrt(pow((2*Absorb_grid-k)/4.,4)+pow((m-(2*(Grid_Nz+4)-2*Absorb_grid))/4.,4))
			 +0.01*sqrt(pow((2*Absorb_grid-k)/4.,12)+pow((m-(2*(Grid_Nz+4)-2*Absorb_grid))/4.,12)));
	}else if(k<=2*Absorb_grid){
	  sigma[k][m]=a*(0.1*pow((2*Absorb_grid-k)/4.,2)
			 +0.01*pow((2*Absorb_grid-k)/4.,6));
	}else if(m<=2*Absorb_grid){
	  sigma[k][m]=a*(0.1*pow((2*Absorb_grid-m)/4.,2)
			 +0.01*pow((2*Absorb_grid-m)/4.,6));
	}else if(m>=2*(Grid_Nz+4)-2*Absorb_grid){
	  sigma[k][m]=a*(0.1*pow((m-(2*(Grid_Nz+4)-2*Absorb_grid))/4.,2)
			 +0.01*pow((m-(2*(Grid_Nz+4)-2*Absorb_grid))/4.,6));
	}
      }
    }
  }else if(myrank==p-1){
    for(k=0;k<2*(Grid_Nx+4);k++){
      for(m=0;m<2*(Grid_Nz+4);m++){
	sigma[k][m]=0.;
	if(k>=2*(Grid_Nx+4)-2*Absorb_grid&&m<=2*Absorb_grid){
	  sigma[k][m]=a*(0.1*sqrt(pow((k-(2*(Grid_Nx+4)-2*Absorb_grid))/4.,4)+pow((2*Absorb_grid-m)/4.,4))
			 +0.01*sqrt(pow((k-(2*(Grid_Nx+4)-2*Absorb_grid))/4.,12)+pow((2*Absorb_grid-m)/4.,12)));
	}else if(k>=2*(Grid_Nx+4)-2*Absorb_grid&&m>=2*(Grid_Nz+4)-2*Absorb_grid){
	  sigma[k][m]=a*(0.1*sqrt(pow((k-(2*(Grid_Nx+4)-2*Absorb_grid))/4.,4)+pow((m-(2*(Grid_Nz+4)-2*Absorb_grid))/4.,4))
			 +0.01*sqrt(pow((k-(2*(Grid_Nx+4)-2*Absorb_grid))/4.,12)+pow((m-(2*(Grid_Nz+4)-2*Absorb_grid))/4.,12)));
	}else if(k>=2*(Grid_Nx+4)-2*Absorb_grid){
	  sigma[k][m]=a*(0.1*pow((k-(2*(Grid_Nx+4)-2*Absorb_grid))/4.,2)
			 +0.01*pow((k-(2*(Grid_Nx+4)-2*Absorb_grid))/4.,6));
	}else if(m<=2*Absorb_grid){
	  sigma[k][m]=a*(0.1*pow((2*Absorb_grid-m)/4.,2)
			 +0.01*pow((2*Absorb_grid-m)/4.,6));
	}else if(m>=2*(Grid_Nz+4)-2*Absorb_grid){
	  sigma[k][m]=a*(0.1*pow((m-(2*(Grid_Nz+4)-2*Absorb_grid))/4.,2)
			 +0.01*pow((m-(2*(Grid_Nz+4)-2*Absorb_grid))/4.,6));
	}
      }
    }
  }else{
    for(k=0;k<2*(Grid_Nx+4);k++){
      for(m=0;m<2*(Grid_Nz+4);m++){
	sigma[k][m]=0.;
	if(m<=2*Absorb_grid){
	  sigma[k][m]=a*(0.1*pow((2*Absorb_grid-m)/4.,2)
			 +0.01*pow((2*Absorb_grid-m)/4.,6));
	}else if(m>=2*(Grid_Nz+4)-2*Absorb_grid){
	  sigma[k][m]=a*(0.1*pow((m-(2*(Grid_Nz+4)-2*Absorb_grid))/4.,2)
			 +0.01*pow((m-(2*(Grid_Nz+4)-2*Absorb_grid))/4.,6));
	}
      }
    }
    }*/

  for(k=2;k<Grid_Nx+2;k++){
    for(m=2;m<Grid_Nz+2;m++){
      x=(myrank*Grid_Nx+(k-2)+0.5)*dx-dx*p*Grid_Nx/2.;
      z=(m-2)*dx-dx*Grid_Nz/2.;
      
      bx=mu0*I/2./Pi*(-(z+R*cos(alpha))
		      /(pow(x+R*sin(alpha),2)+pow(z+R*cos(alpha),2))
		      +(z-R*cos(alpha))
		      /(pow(x-R*sin(alpha),2)+pow(z-R*cos(alpha),2)));

      /*bx=mu0*I*2.*R/2./Pi/pow(pow(x,2)+pow(z,2),2)*
        (-(pow(x,2)-pow(z,2))*cos(alpha)-2.*x*z*sin(alpha));*/


      bx=0.;
      grid[k][m].b0x=bx+IMF_x;
      grid[k][m].bx=bx+IMF_x;
    }
  }

  for(k=2;k<Grid_Nx+2;k++){
    for(m=2;m<Grid_Nz+2;m++){
      x=(myrank*Grid_Nx+(k-2))*dx-dx*p*Grid_Nx/2.;
      z=(m-2+0.5)*dx-dx*Grid_Nz/2.;

      bz=-mu0*I/2./Pi*(-(x+R*sin(alpha))
		      /(pow(x+R*sin(alpha),2)+pow(z+R*cos(alpha),2))
		      +(x-R*sin(alpha))
		      /(pow(x-R*sin(alpha),2)+pow(z-R*cos(alpha),2)));

     /* bz=mu0*I*2.*R/2./Pi/pow(pow(x,2)+pow(z,2),2)*
      (-2.*x*z*cos(alpha)+(pow(x,2)-pow(z,2))*sin(alpha));*/

      bz=0.;
      grid[k][m].b0z=bz+IMF_z;
      grid[k][m].bz=bz+IMF_z;
    }
  }

  return(0);
}

int init_particle(Particle *ion[],Particle *ion_start[],Particle *electron[],Particle *electron_start[],
		  gsl_rng *rnd_p[],gsl_rng *rnd_v[],const int myrank)
{
  int i,j;
  int k,m,thread;
  int flag;
  const gsl_rng_type *T;
  const int loop=N*Step_p/THREAD_N;

  T=gsl_rng_ranlxd2;

  for(i=0;i<THREAD_N;i++){
    rnd_p[i]=gsl_rng_alloc(T);
    rnd_v[i]=gsl_rng_alloc(T);
    //gsl_rng_set(rnd_p[i],(unsigned long int)time(NULL)+i);
    //gsl_rng_set(rnd_v[i],(unsigned long int)time(NULL)+i+THREAD_N);
    gsl_rng_set(rnd_p[i],(i*PROCESS_N+myrank)*100);
    gsl_rng_set(rnd_v[i],(i*PROCESS_N+myrank)*100+50);
  }

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

  flag=0;
  for(i=0;i<THREAD_N;i++){
    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){
      flag++;
      ion[i]->x=gsl_ran_flat(rnd_p[i],2.*dx+i*Grid_Nx*dx/THREAD_N,2.*dx+(i+1)*Grid_Nx*dx/THREAD_N);
      ion[i]->z=gsl_ran_flat(rnd_p[i],2.*dx,2.*dx+Grid_Nz*dx);
      ion[i]->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
      //ion[i]->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
      ion[i]->vy=0.;
      ion[i]->vz=V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
      ion[i]->n=N0i;
      ion[i]->flag=1;

      if(flag==Np*Np){
	flag=0;
	ion[i]->flag=1;
      }

      ion[i]=ion[i]->next_particle;
    }
    flag++;
    ion[i]->x=gsl_ran_flat(rnd_p[i],2.*dx+i*Grid_Nx*dx/THREAD_N,2.*dx+(i+1)*Grid_Nx*dx/THREAD_N);
    ion[i]->z=gsl_ran_flat(rnd_p[i],2.*dx,2.*dx+Grid_Nz*dx);
    ion[i]->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
    //ion[i]->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
    ion[i]->vy=0.;
    ion[i]->vz=V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
    ion[i]->n=N0i;
    ion[i]->flag=1;
    
    if(flag==Np*Np){
      flag=0;
      ion[i]->flag=1;
    }
  }

  flag=0;
  for(i=0;i<THREAD_N;i++){
    electron[i]=electron_start[i];
    while(electron[i]->next_particle!=NULL){
      flag++;
      electron[i]->x=gsl_ran_flat(rnd_p[i],2.*dx+i*Grid_Nx*dx/THREAD_N,2.*dx+(i+1)*Grid_Nx*dx/THREAD_N);
      electron[i]->z=gsl_ran_flat(rnd_p[i],2.*dx,2.*dx+Grid_Nz*dx);
      electron[i]->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
      //electron[i]->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
      electron[i]->vy=0.;
      electron[i]->vz=V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
      electron[i]->n=N0e;
      electron[i]->flag=1;

      if(flag==Np*Np){
	flag=0;
	electron[i]->flag=1;
      }

      electron[i]=electron[i]->next_particle;
    }
    flag++;
    electron[i]->x=gsl_ran_flat(rnd_p[i],2.*dx+i*Grid_Nx*dx/THREAD_N,2.*dx+(i+1)*Grid_Nx*dx/THREAD_N);
    electron[i]->z=gsl_ran_flat(rnd_p[i],2.*dx,2.*dx+Grid_Nz*dx);
    electron[i]->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
    //electron[i]->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
    electron[i]->vy=0.;
    electron[i]->vz=V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
    electron[i]->n=N0e;
    electron[i]->flag=1;
    
    if(flag==Np*Np){
      flag=0;
      electron[i]->flag=1;
    }
  }

 return(0);
}

int fdtd(Grid grid[][Grid_Nz+4],double sigma[][2*(Grid_Nz+4)],const int myrank,const int p,
	 Particle *ion[],Particle *ion_start[],Particle *electron[],Particle *electron_start[],
	 gsl_rng *rnd_p[],gsl_rng *rnd_v[])
{
  int i,j,c;
  int k,m;
  int c_add_ion;
  int c_add_electron;
  int c_output,c_output_f,c_output_p;
  int c_injection;
  const double mmi=mi;
  const double mme=-me;
  const double dti0=dti;
  const double dte0=dte;
  Grid_thread grid_thread[THREAD_N][Grid_Nx+4][Grid_Nz+4];
  int count_i_fwd[THREAD_N];
  int count_i_bak[THREAD_N];
  int count_e_fwd[THREAD_N];
  int count_e_bak[THREAD_N];
  Particle pack_i_fwd[THREAD_N][PACK_N];
  Particle pack_i_bak[THREAD_N][PACK_N];
  Particle pack_e_fwd[THREAD_N][PACK_N];
  Particle pack_e_bak[THREAD_N][PACK_N];
  double Fx,Fz,My,Fx2,Fz2,Fix2,Fiz2,Fx3,Fz3;
  FILE *fp;
  char filename[256],filename2[256],filename3[256];
  time_t time1,time2;

  time(&time1);

  printf("FDTD start in %d/%d\n",myrank,p);

  c_add_ion=0;
  c_add_electron=0;
  c_output=0;
  c_output_f=0;
  c_output_p=0;
  c_injection=0;

  if(myrank==p/2-1){
    sprintf(filename,"pic%d-wt.txt",version);
    fp=fopen(filename,"w");
    fclose(fp);
  }

  if(myrank==p/2-2){
    sprintf(filename2,"pic%d-wt2.txt",version);
    fp=fopen(filename2,"w");
    fclose(fp);
  }

  if(myrank==p/2+1){
    sprintf(filename3,"pic%d-wt3.txt",version);
    fp=fopen(filename3,"w");
    fclose(fp);
  }
 
  renew_grid(grid,grid_thread);

#pragma omp parallel for  
  for(i=0;i<THREAD_N;i++){
    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){	
      if(ion[i]->x<(Grid_Nx+3.)*dx&&ion[i]->z<(Grid_Nz+2.)*dx&&ion[i]->x>=dx&&ion[i]->z>=2.*dx){
	shape_func_ion0_n(ion[i],i,grid_thread);
      }
      ion[i]=ion[i]->next_particle;
    }
    if(ion[i]->x<(Grid_Nx+3.)*dx&&ion[i]->z<(Grid_Nz+2.)*dx&&ion[i]->x>=dx&&ion[i]->z>=2.*dx){
	shape_func_ion0_n(ion[i],i,grid_thread);
    }
    
    electron[i]=electron_start[i];
    while(electron[i]->next_particle!=NULL){	
      if(electron[i]->x<(Grid_Nx+3.)*dx&&electron[i]->z<(Grid_Nz+2.)*dx&&electron[i]->x>=dx&&electron[i]->z>=2.*dx){
	shape_func_electron0_n(electron[i],i,grid_thread);
      }
      electron[i]=electron[i]->next_particle;
    }
    if(electron[i]->x<(Grid_Nx+3.)*dx&&electron[i]->z<(Grid_Nz+2.)*dx&&electron[i]->x>=dx&&electron[i]->z>=2.*dx){
      shape_func_electron0_n(electron[i],i,grid_thread);
    }
  }
  
  for(i=0;i<THREAD_N;i++){
    for(k=0;k<Grid_Nx+4;k++){
      for(m=0;m<Grid_Nz+4;m++){
	grid[k][m].ni+=grid_thread[i][k][m].ni;
	grid[k][m].ne+=grid_thread[i][k][m].ne;
      }
    }
  }
  
  sr_density(grid,myrank,p);
  
  sr_grid(grid,myrank,p);
  
  renew_phi(grid,myrank,p,0);
  
  sr_grid(grid,myrank,p);
  
  MPI_Barrier(MPI_COMM_WORLD);

  for(c=0;c<Step;c++){
    renew_grid(grid,grid_thread);

    if(c_output>999){//999
      c_output=0;
    }

    if(c_output_f>499){//499
      c_output_f=0;
    }

    if(c_output_p>999){//3124//249
      c_output_p=0;
    }

    if(c_add_ion>249){
      add_ion5(ion,ion_start,rnd_p,rnd_v,myrank,p);
      c_add_ion=0;
    }

    if(c_add_electron>249){
      add_electron5(electron,electron_start,rnd_p,rnd_v,myrank,p);
      c_add_electron=0;
    }

    /*if(c>50000&&c_injection>99){
      add_ion4(ion,ion_start,rnd_p,rnd_v,myrank,p);
      add_electron4(electron,electron_start,rnd_p,rnd_v,myrank,p);
      c_injection=0;
      }*/

    renew_b4(grid,sigma);

    sr_grid(grid,myrank,p);
  
#pragma omp parallel for  
    for(i=0;i<THREAD_N;i++){
      ion[i]=ion_start[i];
      while(ion[i]->next_particle!=NULL){
	if(ion[i]->z<Absorb_grid2*dx||ion[i]->z>(Grid_Nz+4-Absorb_grid2)*dx
	   ||(myrank==0&&ion[i]->x<Absorb_grid2*dx)||(myrank==p-1&&ion[i]->x>(Grid_Nx+4-Absorb_grid2)*dx)){
	  ion[i]->x+=ion[i]->vx*dti0*0.5;
	  ion[i]->z+=ion[i]->vz*dti0*0.5;
	}else{
	  cal_track_c4(ion[i],&mmi,&dti0,grid);
	}

	if(ion[i]->x<(Grid_Nx+3.)*dx&&ion[i]->x>=dx&&ion[i]->z<(Grid_Nz+2.)*dx&&ion[i]->z>=2.*dx){
	  shape_func_ion0_jy(ion[i],i,grid_thread);
	  shape_func_ion0_jx(ion[i],i,grid_thread);
	  shape_func_ion0_jz(ion[i],i,grid_thread);
	}
	ion[i]=ion[i]->next_particle;
      }
      if(ion[i]->z<Absorb_grid2*dx||ion[i]->z>(Grid_Nz+4-Absorb_grid2)*dx
	 ||(myrank==0&&ion[i]->x<Absorb_grid2*dx)||(myrank==p-1&&ion[i]->x>(Grid_Nx+4-Absorb_grid2)*dx)){
	ion[i]->x+=ion[i]->vx*dti0*0.5;
	ion[i]->z+=ion[i]->vz*dti0*0.5;
      }else{
	cal_track_c4(ion[i],&mmi,&dti0,grid);
      }
      
      if(ion[i]->x<(Grid_Nx+3.)*dx&&ion[i]->x>=dx&&ion[i]->z<(Grid_Nz+2.)*dx&&ion[i]->z>=2.*dx){
	shape_func_ion0_jy(ion[i],i,grid_thread);
	shape_func_ion0_jx(ion[i],i,grid_thread);
	shape_func_ion0_jz(ion[i],i,grid_thread);
      }

      electron[i]=electron_start[i];
      while(electron[i]->next_particle!=NULL){	
	if(electron[i]->z<Absorb_grid2*dx||electron[i]->z>(Grid_Nz+4-Absorb_grid2)*dx
	   ||(myrank==0&&electron[i]->x<Absorb_grid2*dx)||(myrank==p-1&&electron[i]->x>(Grid_Nx+4-Absorb_grid2)*dx)){
	  electron[i]->x+=electron[i]->vx*dte0*0.5;
	  electron[i]->z+=electron[i]->vz*dte0*0.5;
	}else{
	  cal_track_c4(electron[i],&mme,&dte0,grid);
	}

	if(electron[i]->x<(Grid_Nx+3.)*dx&&electron[i]->x>=dx&&electron[i]->z<(Grid_Nz+2.)*dx&&electron[i]->z>=2.*dx){
	  shape_func_electron0_jy(electron[i],i,grid_thread);
	  shape_func_electron0_jx(electron[i],i,grid_thread);
	  shape_func_electron0_jz(electron[i],i,grid_thread);
	}
	electron[i]=electron[i]->next_particle;
      }
      if(electron[i]->z<Absorb_grid2*dx||electron[i]->z>(Grid_Nz+4-Absorb_grid2)*dx
	 ||(myrank==0&&electron[i]->x<Absorb_grid2*dx)||(myrank==p-1&&electron[i]->x>(Grid_Nx+4-Absorb_grid2)*dx)){
	electron[i]->x+=electron[i]->vx*dte0*0.5;
	electron[i]->z+=electron[i]->vz*dte0*0.5;
      }else{
	cal_track_c4(electron[i],&mme,&dte0,grid);
      }
      
      if(electron[i]->x<(Grid_Nx+3.)*dx&&electron[i]->x>=dx&&electron[i]->z<(Grid_Nz+2.)*dx&&electron[i]->z>=2.*dx){
	shape_func_electron0_jy(electron[i],i,grid_thread);
	shape_func_electron0_jx(electron[i],i,grid_thread);
	shape_func_electron0_jz(electron[i],i,grid_thread);
      }
    }

#pragma unroll
    for(i=0;i<THREAD_N;i++){
      for(k=0;k<Grid_Nx+4;k++){
	for(m=0;m<Grid_Nz+4;m++){
	  grid[k][m].jix0+=grid_thread[i][k][m].jix0;
	  grid[k][m].jiy0+=grid_thread[i][k][m].jiy0;
	  grid[k][m].jiz0+=grid_thread[i][k][m].jiz0;
	  grid[k][m].jex0+=grid_thread[i][k][m].jex0;
	  grid[k][m].jey0+=grid_thread[i][k][m].jey0;
	  grid[k][m].jez0+=grid_thread[i][k][m].jez0;
	}
      }
    }

    sr_current(grid,myrank,p);

    external_current_perp(grid,c,myrank,p);
    
    renew_b4(grid,sigma);

    sr_grid(grid,myrank,p);

#pragma omp parallel for  
    for(i=0;i<THREAD_N;i++){
      ion[i]=ion_start[i];
      while(ion[i]->next_particle!=NULL){
	ion[i]->x+=ion[i]->vx*dti0*0.5;
	ion[i]->z+=ion[i]->vz*dti0*0.5;
	
	if(ion[i]->x<(Grid_Nx+3.)*dx&&ion[i]->x>=dx&&ion[i]->z<(Grid_Nz+2.)*dx&&ion[i]->z>=2.*dx){
	  shape_func_ion0_n(ion[i],i,grid_thread);
	}
	ion[i]=ion[i]->next_particle;
      }
      ion[i]->x+=ion[i]->vx*dti0*0.5;
      ion[i]->z+=ion[i]->vz*dti0*0.5;
      
      if(ion[i]->x<(Grid_Nx+3.)*dx&&ion[i]->x>=dx&&ion[i]->z<(Grid_Nz+2.)*dx&&ion[i]->z>=2.*dx){
	shape_func_ion0_n(ion[i],i,grid_thread);
      }

      electron[i]=electron_start[i];
      while(electron[i]->next_particle!=NULL){	
	electron[i]->x+=electron[i]->vx*dte0*0.5;
	electron[i]->z+=electron[i]->vz*dte0*0.5;
	
	if(electron[i]->x<(Grid_Nx+3.)*dx&&electron[i]->x>=dx&&electron[i]->z<(Grid_Nz+2.)*dx&&electron[i]->z>=2.*dx){
	  shape_func_electron0_n(electron[i],i,grid_thread);
	}
	electron[i]=electron[i]->next_particle;
      }
      electron[i]->x+=electron[i]->vx*dte0*0.5;
      electron[i]->z+=electron[i]->vz*dte0*0.5;
      
      if(electron[i]->x<(Grid_Nx+3.)*dx&&electron[i]->x>=dx&&electron[i]->z<(Grid_Nz+2.)*dx&&electron[i]->z>=2.*dx){
	shape_func_electron0_n(electron[i],i,grid_thread);
      }
      electron[i]=electron[i]->next_particle;
    }

#pragma unroll
    for(i=0;i<THREAD_N;i++){
      for(k=0;k<Grid_Nx+4;k++){
	for(m=0;m<Grid_Nz+4;m++){
	  grid[k][m].ni+=grid_thread[i][k][m].ni;
	  grid[k][m].ne+=grid_thread[i][k][m].ne;
	}
      }
    }

    sr_density(grid,myrank,p);

    renew_e4(grid,sigma);

    sr_grid(grid,myrank,p);

    renew_phi(grid,myrank,p,c);

    sr_grid(grid,myrank,p);

    /////MPI_Barrier(MPI_COMM_WORLD);

#pragma omp parallel for
    for(i=0;i<THREAD_N;i++){
      count_i_fwd[i]=0;
      count_i_bak[i]=0;
      count_e_fwd[i]=0;
      count_e_bak[i]=0;
    }

#pragma omp parallel for  
    for(i=0;i<THREAD_N;i++){
      ion[i]=ion_start[i];
      while(ion[i]->next_particle!=NULL){
	if(ion[i]->x<(Grid_Nx+3.)*dx&&ion[i]->x>=(Grid_Nx+2.)*dx&&ion[i]->z<(Grid_Nz+2.)*dx&&ion[i]->z>=2.*dx){
	  pack_i_fwd[i][count_i_fwd[i]]=*ion[i];
	  count_i_fwd[i]++;
	}else if(ion[i]->x>dx&&ion[i]->x<2.*dx&&ion[i]->z<(Grid_Nz+2.)*dx&&ion[i]->z>=2.*dx){
	  pack_i_bak[i][count_i_bak[i]]=*ion[i];
	  count_i_bak[i]++;
	}
	ion[i]=ion[i]->next_particle;
      }
      if(ion[i]->x<(Grid_Nx+3.)*dx&&ion[i]->x>=(Grid_Nx+2.)*dx&&ion[i]->z<(Grid_Nz+2.)*dx&&ion[i]->z>=2.*dx){
	pack_i_fwd[i][count_i_fwd[i]]=*ion[i];
	count_i_fwd[i]++;
      }else if(ion[i]->x>dx&&ion[i]->x<2.*dx&&ion[i]->z<(Grid_Nz+2.)*dx&&ion[i]->z>=2.*dx){
	pack_i_bak[i][count_i_bak[i]]=*ion[i];
	count_i_bak[i]++;
      }

      electron[i]=electron_start[i];
      while(electron[i]->next_particle!=NULL){	
	if(electron[i]->x<(Grid_Nx+3.)*dx&&electron[i]->x>=(Grid_Nx+2.)*dx&&electron[i]->z<(Grid_Nz+2.)*dx&&electron[i]->z>=2.*dx){
	   pack_e_fwd[i][count_e_fwd[i]]=*electron[i];
	   count_e_fwd[i]++;
	}else if(electron[i]->x>dx&&electron[i]->x<2.*dx&&electron[i]->z<(Grid_Nz+2.)*dx&&electron[i]->z>=2.*dx){
	   pack_e_bak[i][count_e_bak[i]]=*electron[i];
	   count_e_bak[i]++;
	}
	electron[i]=electron[i]->next_particle;
      }
      if(electron[i]->x<(Grid_Nx+3.)*dx&&electron[i]->x>=(Grid_Nx+2.)*dx&&electron[i]->z<(Grid_Nz+2.)*dx&&electron[i]->z>=2.*dx){
	pack_e_fwd[i][count_e_fwd[i]]=*electron[i];
	count_e_fwd[i]++;
      }else if(electron[i]->x>dx&&electron[i]->x<2.*dx&&electron[i]->z<(Grid_Nz+2.)*dx&&electron[i]->z>=2.*dx){
	pack_e_bak[i][count_e_bak[i]]=*electron[i];
	count_e_bak[i]++;
      }
    }

    if(myrank==0){
      for(i=0;i<THREAD_N;i++){
	count_i_bak[i]=0;
	count_e_bak[i]=0;
      }
    }else if(myrank==p-1){
      for(i=0;i<THREAD_N;i++){
	count_i_fwd[i]=0;
	count_e_fwd[i]=0;
      }
    }

    sr_particle(pack_i_fwd,count_i_fwd,ion,ion_start,myrank,p,1);
    sr_particle(pack_i_bak,count_i_bak,ion,ion_start,myrank,p,-1);
    sr_particle(pack_e_fwd,count_e_fwd,electron,electron_start,myrank,p,1);
    sr_particle(pack_e_bak,count_e_bak,electron,electron_start,myrank,p,-1);

#pragma omp parallel for  
    for(i=0;i<THREAD_N;i++){
      ion[i]=ion_start[i];
      while(ion[i]->next_particle!=NULL){
	if(ion[i]->x<2.*dx||ion[i]->x>=(Grid_Nx+2.)*dx||ion[i]->z>=(Grid_Nz+2.)*dx||ion[i]->z<2.*dx){
	  del_particle_i(ion,i,ion_start);
	}
	ion[i]=ion[i]->next_particle;
      }
      if(ion[i]->x<2.*dx||ion[i]->x>=(Grid_Nx+2.)*dx||ion[i]->z>=(Grid_Nz+2.)*dx||ion[i]->z<2.*dx){
	del_particle_i(ion,i,ion_start);
      }

      electron[i]=electron_start[i];
      while(electron[i]->next_particle!=NULL){	
	if(electron[i]->x<2.*dx||electron[i]->x>=(Grid_Nx+2.)*dx||electron[i]->z>=(Grid_Nz+2.)*dx||electron[i]->z<2.*dx){
	  del_particle_e(electron,i,electron_start);
	}
	electron[i]=electron[i]->next_particle;
      }
      if(electron[i]->x<2.*dx||electron[i]->x>=(Grid_Nx+2.)*dx||electron[i]->z>=(Grid_Nz+2.)*dx||electron[i]->z<2.*dx){
	del_particle_e(electron,i,electron_start);
      }
    }

    if(c_output_f==0){
      thrust_F(grid,myrank,p,&Fx,&Fz,&My);
      thrust_F2(grid,myrank,p,&Fx2,&Fz2);
      thrust_F3(grid,myrank,p,&Fx3,&Fz3);

      if(myrank==0){
	time(&time2);
	fprintf(stderr,"%f %E %E %E\n",difftime(time2,time1),Fx,Fz,My);
      }
    }

    if(c_output_p==0){
      // output_p(ion,ion_start,electron,electron_start,c,myrank,p);
    }

    if(c_output==0){  
      output(grid,c,myrank,p);
    }


    /* output_wave_data*/
    if(myrank==p/2-1){
      fp=fopen(filename,"a");
      k=2;
      m=226;  //x=0 z=-900
      fprintf(fp,"%E %E %E %E %E %E %E %E\n",
	      grid[k][m].ex,
	      grid[k][m].ey,
	      grid[k][m].ez,
	      grid[k][m].bx,
	      grid[k][m].by,
	      grid[k][m].bz,
	      fabs(grid[k][m].bx-grid[k][m].b0x
	      -grid[k-1][m].bx+grid[k-1][m].b0x
	      +grid[k][m].bz-grid[k][m].b0z
	       -grid[k][m-1].bz+grid[k][m-1].b0z)
	      /dx/sqrt(pow(grid[k][m].bx+grid[k-1][m].bx,2)/4.
		       +pow(grid[k][m].by,2)
		       +pow(grid[k][m].bz+grid[k][m-1].bz,2)/4.),
	      fabs((grid[k+1][m].ex-grid[k][m].ex
		+grid[k][m+1].ez-grid[k][m].ez)
	       /dx-q*(grid[k][m].ni-grid[k][m].ne)/e0)
	      /sqrt(pow(grid[k+1][m].ex+grid[k][m].ex,2)/4.
		    +pow(grid[k][m].ey,2)
		    +pow(grid[k][m+1].ez+grid[k][m].ez,2)/4.)
	      );
      fclose(fp);
    }

    if(myrank==p/2-2){
      fp=fopen(filename2,"a");
      k=15;
      m=256;  //x=-1200 z=0
      fprintf(fp,"%E %E %E %E %E %E %E %E\n",
	      grid[k][m].ex,
	      grid[k][m].ey,
	      grid[k][m].ez,
	      grid[k][m].bx,
	      grid[k][m].by,
	      grid[k][m].bz,
	      fabs(grid[k][m].bx-grid[k][m].b0x
	      -grid[k-1][m].bx+grid[k-1][m].b0x
	      +grid[k][m].bz-grid[k][m].b0z
	       -grid[k][m-1].bz+grid[k][m-1].b0z)
	      /dx/sqrt(pow(grid[k][m].bx+grid[k-1][m].bx,2)/4.
		       +pow(grid[k][m].by,2)
		       +pow(grid[k][m].bz+grid[k][m-1].bz,2)/4.),
	      fabs((grid[k+1][m].ex-grid[k][m].ex
		+grid[k][m+1].ez-grid[k][m].ez)
	       /dx-q*(grid[k][m].ni-grid[k][m].ne)/e0)
	      /sqrt(pow(grid[k+1][m].ex+grid[k][m].ex,2)/4.
		    +pow(grid[k][m].ey,2)
		    +pow(grid[k][m+1].ez+grid[k][m].ez,2)/4.)
	      );
      fclose(fp);
    }

    if(myrank==p/2+1){
      fp=fopen(filename3,"a");
      k=10;
      m=256;  //x=1200 z=0
      fprintf(fp,"%E %E %E %E %E %E %E %E\n",
	      grid[k][m].ex,
	      grid[k][m].ey,
	      grid[k][m].ez,
	      grid[k][m].bx,
	      grid[k][m].by,
	      grid[k][m].bz,
	      fabs(grid[k][m].bx-grid[k][m].b0x
	      -grid[k-1][m].bx+grid[k-1][m].b0x
	      +grid[k][m].bz-grid[k][m].b0z
	       -grid[k][m-1].bz+grid[k][m-1].b0z)
	      /dx/sqrt(pow(grid[k][m].bx+grid[k-1][m].bx,2)/4.
		       +pow(grid[k][m].by,2)
		       +pow(grid[k][m].bz+grid[k][m-1].bz,2)/4.),
	      fabs((grid[k+1][m].ex-grid[k][m].ex
		+grid[k][m+1].ez-grid[k][m].ez)
	       /dx-q*(grid[k][m].ni-grid[k][m].ne)/e0)
	      /sqrt(pow(grid[k+1][m].ex+grid[k][m].ex,2)/4.
		    +pow(grid[k][m].ey,2)
		    +pow(grid[k][m+1].ez+grid[k][m].ez,2)/4.)
	      );
      fclose(fp);
    }


    c_add_ion++;
    c_add_electron++;
    c_output++;
    c_output_f++;
    c_output_p++;
    c_injection++;

  }

  return(0);
}

int sr_grid(Grid grid[][Grid_Nz+4],const int myrank,const int p)
{
  int src,dest,tag,count;
  MPI_Status stat;
  MPI_Request request;
  int i,j;

  tag=1000;

  /////MPI_Barrier(MPI_COMM_WORLD);

  count=2*(Grid_Nz+4)*sizeof(Grid)/sizeof(double);

  src=myrank-1;

  if(src==-1){
    src=p-1;
  }

  MPI_Irecv(&grid[0],count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=myrank+1;

  if(dest==p){
    dest=0;
  }

  MPI_Send(&grid[Grid_Nx],count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  src=myrank+1;

  if(src==p){
    src=0;
  }

  MPI_Irecv(&grid[Grid_Nx+2],count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=myrank-1;

  if(dest==-1){
    dest=p-1;
  }

  MPI_Send(&grid[2],count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

  return(0);
}

int sr_current(Grid grid[][Grid_Nz+4],const int myrank,const int p)
{
  int src,dest,tag,count;
  MPI_Status stat;
  MPI_Request request;
  int i,j;
  static Grid grid_r[Grid_Nz+4];

  tag=1000;

  ////////MPI_Barrier(MPI_COMM_WORLD);

  count=(Grid_Nz+4)*sizeof(Grid)/sizeof(double);

  src=myrank-1;

  if(src==-1){
    src=p-1;
  }

  MPI_Irecv(&grid_r,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=myrank+1;

  if(dest==p){
    dest=0;
  }

  MPI_Send(&grid[Grid_Nx+2],count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

#pragma unroll
  for(i=0;i<Grid_Nz+4;i++){
    grid[2][i].jix0+=grid_r[i].jix0;
    grid[2][i].jiy0+=grid_r[i].jiy0;
    grid[2][i].jiz0+=grid_r[i].jiz0;
    grid[2][i].jex0+=grid_r[i].jex0;
    grid[2][i].jey0+=grid_r[i].jey0;
    grid[2][i].jez0+=grid_r[i].jez0;
  }

  src=myrank+1;

  if(src==p){
    src=0;
  }

  MPI_Irecv(&grid_r,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=myrank-1;

  if(dest==-1){
    dest=p-1;
  }

  MPI_Send(&grid[1],count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

#pragma unroll
  for(i=0;i<Grid_Nz+4;i++){
    grid[Grid_Nx+1][i].jix0+=grid_r[i].jix0;
    grid[Grid_Nx+1][i].jiy0+=grid_r[i].jiy0;
    grid[Grid_Nx+1][i].jiz0+=grid_r[i].jiz0;
    grid[Grid_Nx+1][i].jex0+=grid_r[i].jex0;
    grid[Grid_Nx+1][i].jey0+=grid_r[i].jey0;
    grid[Grid_Nx+1][i].jez0+=grid_r[i].jez0;
  }

  return(0);
}

int sr_density(Grid grid[][Grid_Nz+4],const int myrank,const int p)
{
  int src,dest,tag,count;
  MPI_Status stat;
  MPI_Request request;
  int i,j;
  static Grid grid_r[Grid_Nz+4];

  tag=1000;

  //////MPI_Barrier(MPI_COMM_WORLD);

  count=(Grid_Nz+4)*sizeof(Grid)/sizeof(double);

  src=myrank-1;

  if(src==-1){
    src=p-1;
  }

  MPI_Irecv(&grid_r,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=myrank+1;

  if(dest==p){
    dest=0;
  }

  MPI_Send(&grid[Grid_Nx+2],count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

#pragma unroll
  for(i=0;i<Grid_Nz+4;i++){
    grid[2][i].ni+=grid_r[i].ni;
    grid[2][i].ne+=grid_r[i].ne;
  }

  src=myrank+1;

  if(src==p){
    src=0;
  }

  MPI_Irecv(&grid_r,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request);

  dest=myrank-1;

  if(dest==-1){
    dest=p-1;
  }

  MPI_Send(&grid[1],count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

  MPI_Wait(&request,&stat);

#pragma unroll
  for(i=0;i<Grid_Nz+4;i++){
    grid[Grid_Nx+1][i].ni+=grid_r[i].ni;
    grid[Grid_Nx+1][i].ne+=grid_r[i].ne;
  }

  return(0);
}

int sr_particle(const Particle pack_particle[][PACK_N],int count_p[],Particle *particle[],Particle *particle_start[],const int myrank,const int p,const int flag)
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

  //////MPI_Barrier(MPI_COMM_WORLD);

  if(flag==1){
    for(i=0;i<THREAD_N;i++){
      src=myrank-1;
      
      if(src==-1){
	src=p-1;
      }
      
      MPI_Irecv(&count_p_r[i],1,MPI_INT,src,tag,MPI_COMM_WORLD,&request[i]);
      
      dest=myrank+1;
      
      if(dest==p){
	dest=0;
      }
      
      MPI_Send(&count_p[i],1,MPI_INT,dest,tag,MPI_COMM_WORLD);
      
      MPI_Wait(&request[i],&stat[i]);
    }

    for(i=0;i<THREAD_N;i++){
      count=count_p_r[i]*sizeof(Particle)/sizeof(int);
      
      src=myrank-1;
      
      if(src==-1){
	src=p-1;
      }
      
      MPI_Irecv(&pack_particle_r[i],count,MPI_INT,src,tag,MPI_COMM_WORLD,&request[i]);

      count=count_p[i]*sizeof(Particle)/sizeof(int);
      dest=myrank+1;
      
      if(dest==p){
	dest=0;
      }
      
      MPI_Send(&pack_particle[i],count,MPI_INT,dest,tag,MPI_COMM_WORLD);
      
      MPI_Wait(&request[i],&stat[i]);
    }
  }else if(flag==-1){
    for(i=0;i<THREAD_N;i++){
      src=myrank+1;
      
      if(src==p){
	src=0;
      }
      
      MPI_Irecv(&count_p_r[i],1,MPI_INT,src,tag,MPI_COMM_WORLD,&request[i]);

      dest=myrank-1;
      
      if(dest==-1){
	dest=p-1;
      }
      
      MPI_Send(&count_p[i],1,MPI_INT,dest,tag,MPI_COMM_WORLD);
      
      MPI_Wait(&request[i],&stat[i]);
    }

    for(i=0;i<THREAD_N;i++){
      count=count_p_r[i]*sizeof(Particle)/sizeof(int);

      src=myrank+1;
    
      if(src==p){
	src=0;
      }

      MPI_Irecv(&pack_particle_r[i],count,MPI_INT,src,tag,MPI_COMM_WORLD,&request[i]);

      count=count_p[i]*sizeof(Particle)/sizeof(int);
      dest=myrank-1;
      
      if(dest==-1){
	dest=p-1;
      }
 
      MPI_Send(&pack_particle[i],count,MPI_INT,dest,tag,MPI_COMM_WORLD);

      MPI_Wait(&request[i],&stat[i]);
    }
  }

#pragma omp parallel for private(j)
  for(i=0;i<THREAD_N;i++){
    particle[i]=particle_start[i];
    while(particle[i]->next_particle!=NULL){
      particle[i]=particle[i]->next_particle;
    }

    for(j=0;j<count_p_r[i];j++){
      if((particle[i]->next_particle=(Particle*)malloc(sizeof(Particle)))!=NULL){
	if(flag==1){
	  particle[i]->next_particle->x=pack_particle_r[i][j].x-Grid_Nx*dx;
	}else if(flag==-1){
	  particle[i]->next_particle->x=pack_particle_r[i][j].x+Grid_Nx*dx;
	}
   
	particle[i]->next_particle->z=pack_particle_r[i][j].z;
	particle[i]->next_particle->vx=pack_particle_r[i][j].vx;
	particle[i]->next_particle->vy=pack_particle_r[i][j].vy;
	particle[i]->next_particle->vz=pack_particle_r[i][j].vz;
	particle[i]->next_particle->n=pack_particle_r[i][j].n;
	particle[i]->next_particle->flag=pack_particle_r[i][j].flag;

	particle[i]->next_particle->prev_particle=particle[i];
	particle[i]=particle[i]->next_particle;
	particle[i]->next_particle=NULL;
      }
    }
  }
  
  return(0);
}

int renew_grid(Grid grid[][Grid_Nz+4],Grid_thread grid_thread[][Grid_Nx+4][Grid_Nz+4])
{
  int i,k,m;

#pragma omp parallel for private(m)
  for(k=0;k<Grid_Nx+4;k++){
    for(m=0;m<Grid_Nz+4;m++){    
      grid[k][m].jix0=0.;
      grid[k][m].jiy0=0.;
      grid[k][m].jiz0=0.;
      
      grid[k][m].jex0=0.;
      grid[k][m].jey0=0.;
      grid[k][m].jez0=0.;
      
      grid[k][m].ni=0.;
      grid[k][m].ne=0.;

      grid[k][m].phi=0.;
    }
  }

#pragma omp parallel for private(k,m)
  for(i=0;i<THREAD_N;i++){
    for(k=0;k<Grid_Nx+4;k++){
      for(m=0;m<Grid_Nz+4;m++){    
	grid_thread[i][k][m].jix0=0.;
	grid_thread[i][k][m].jiy0=0.;
	grid_thread[i][k][m].jiz0=0.;
	
	grid_thread[i][k][m].jex0=0.;
	grid_thread[i][k][m].jey0=0.;
	grid_thread[i][k][m].jez0=0.;
	
	grid_thread[i][k][m].ni=0.;
	grid_thread[i][k][m].ne=0.;
      }
    }
  }
    
  return(0);
}

int renew_b4(Grid grid[][Grid_Nz+4],const double sigma[][2*(Grid_Nz+4)])
{
  int k,m;
  
#pragma omp parallel for private(m)
  for(k=2;k<Grid_Nx+2;k++){
    for(m=2;m<Grid_Nz+2;m++){
      grid[k][m].bx=grid[k][m].b0x
      	+(1.-sigma[2*k+1][2*m]/e0*dtf*0.5*0.5)/(1.+sigma[2*k+1][2*m]/e0*dtf*0.5*0.5)*(grid[k][m].bx-grid[k][m].b0x)
	+0.5*dtf/(1.+sigma[2*k+1][2*m]/e0*dtf*0.5*0.5)
	*1./24.*(27.*(grid[k][m].ey-grid[k][m-1].ey)-(grid[k][m+1].ey-grid[k][m-2].ey))/dx;
      
      grid[k][m].by=grid[k][m].b0y
	+(1.-sigma[2*k][2*m]/e0*dtf*0.5*0.5)/(1.+sigma[2*k][2*m]/e0*dtf*0.5*0.5)*(grid[k][m].by-grid[k][m].b0y)
	-dtf*0.5/(1.+sigma[2*k][2*m]/e0*dtf*0.5*0.5)
	*(1./24.*(27.*(grid[k][m].ex-grid[k][m-1].ex)-(grid[k][m+1].ex-grid[k][m-2].ex))/dx
	  -1./24.*(27.*(grid[k][m].ez-grid[k-1][m].ez)-(grid[k+1][m].ez-grid[k-2][m].ez))/dx);
      
      grid[k][m].bz=grid[k][m].b0z
	+(1.-sigma[2*k][2*m+1]/e0*dtf*0.5*0.5)/(1.+sigma[2*k][2*m+1]/e0*dtf*0.5*0.5)*(grid[k][m].bz-grid[k][m].b0z)
	-dtf*0.5/(1.+sigma[2*k][2*m+1]/e0*dtf*0.5*0.5)
	*1./24.*(27.*(grid[k][m].ey-grid[k-1][m].ey)-(grid[k+1][m].ey-grid[k-2][m].ey))/dx;
    }
  }
  
  return(0);
}

int renew_e4(Grid grid[][Grid_Nz+4],const double sigma[][2*(Grid_Nz+4)])
{
  int k,m;

#pragma omp parallel for private(m)
  for(k=2;k<Grid_Nx+2;k++){
    for(m=2;m<Grid_Nz+2;m++){
      grid[k][m].ex=(1.-0.5*mu0*sigma[2*k][2*m+1]*C*C*dtf)/(1.+0.5*mu0*sigma[2*k][2*m+1]*C*C*dtf)*grid[k][m].ex
	+C*C*dtf/(1.+0.5*mu0*sigma[2*k][2*m+1]*C*C*dtf)
	*(-1./24.*(27.*(grid[k][m+1].by-grid[k][m+1].b0y-grid[k][m].by+grid[k][m].b0y)
		   -(grid[k][m+2].by-grid[k][m+2].b0y-grid[k][m-1].by+grid[k][m-1].b0y))/dx
	  -mu0*(grid[k][m].jix0-grid[k][m].jex0));//
  
      grid[k][m].ez=(1.-0.5*mu0*sigma[2*k+1][2*m]*C*C*dtf)/(1.+0.5*mu0*sigma[2*k+1][2*m]*C*C*dtf)*grid[k][m].ez
	+C*C*dtf/(1.+0.5*mu0*sigma[2*k+1][2*m]*C*C*dtf)
	*(1./24.*(27.*(grid[k+1][m].by-grid[k+1][m].b0y-grid[k][m].by+grid[k][m].b0y)
		  -(grid[k+2][m].by-grid[k+2][m].b0y-grid[k-1][m].by+grid[k-1][m].b0y))/dx
	  -mu0*(grid[k][m].jiz0-grid[k][m].jez0));///
  
      grid[k][m].ey=(1.-0.5*mu0*sigma[2*k+1][2*m+1]*C*C*dtf)/(1.+0.5*mu0*sigma[2*k+1][2*m+1]*C*C*dtf)*grid[k][m].ey
	+C*C*dtf/(1.+0.5*mu0*sigma[2*k+1][2*m+1]*C*C*dtf)
	*(1./24.*(27.*(grid[k][m+1].bx-grid[k][m+1].b0x-grid[k][m].bx+grid[k][m].b0x)
		  -(grid[k][m+2].bx-grid[k][m+2].b0x-grid[k][m-1].bx+grid[k][m-1].b0x))/dx
	  -1./24.*(27.*(grid[k+1][m].bz-grid[k+1][m].b0z-grid[k][m].bz+grid[k][m].b0z)
		   -(grid[k+2][m].bz-grid[k+2][m].b0z-grid[k-1][m].bz+grid[k-1][m].b0z))/dx
	  -mu0*(grid[k][m].jiy0-grid[k][m].jey0));//
    }
  }

  return(0);
}

inline int cal_track_c3(Particle *p,double *m,double *dt,Grid grid[][Grid_Nz+4])
{
  double ux,uy,uz;
  double uox,uoy,uoz;
  double upx,upy,upz;
  double bux,buy,buz;
  double gamma1,gamma2;
  double ex,ey,ez,bx,by,bz;
  const double T=0.5*q*(*dt)/(*m);
  double S;

  shape_func_ex_bz3(p,&ex,&bz,grid);
  shape_func_ey3(p,&ey,grid);
  shape_func_ez_bx3(p,&ez,&bx,grid);
  shape_func_by3(p,&by,grid);

  ex+=V*IMF_y;
  ey+=-V*IMF_x;

  gamma1=C/sqrt(C*C-p->vx*p->vx-p->vy*p->vy-p->vz*p->vz);

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

  /*if(isnan(gamma1)){
    fprintf(stderr,"Verosity >C\n %f %f %f\n %f\n",p->vx,p->vy,p->vz,gamma2);
    exit(-1);
    }*/

  p->x+=p->vx*(*dt)*0.5;
  p->z+=p->vz*(*dt)*0.5;

  p->n=p->n;
  p->flag=p->flag;

  return(0);
}

inline int cal_track_c4(Particle *p,const double *m,const double *dt,const Grid grid[][Grid_Nz+4])
{
  double ux,uy,uz;
  double uox,uoy,uoz;
  double upx,upy,upz;
  double ex,ey,ez,bx,by,bz;
  const double T=0.5*q*(*dt)/(*m);
  double S;

  shape_func_ex_bz3(p,&ex,&bz,grid);
  shape_func_ey3(p,&ey,grid);
  shape_func_ez_bx3(p,&ez,&bx,grid);
  shape_func_by3(p,&by,grid);

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

  p->x+=p->vx*(*dt)*0.5;
  p->z+=p->vz*(*dt)*0.5;

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
    ion_start[i]=ion[i]->next_particle;
    free(ion[i]->next_particle->prev_particle);
    ion[i]->next_particle->prev_particle=NULL;
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
    electron_start[i]=electron[i]->next_particle;
    free(electron[i]->next_particle->prev_particle);
    electron[i]->next_particle->prev_particle=NULL;
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

int add_ion3(Particle *ion[],Particle *ion_start[],gsl_rng *rnd_p[],gsl_rng *rnd_v[],const int myrank,const int mpi)
{
  int i,j;
  int flag;
  Particle *p,*p1;

#pragma omp parallel for private(p,p1,j)
  for(i=0;i<THREAD_N;i++){
    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){
      ion[i]=ion[i]->next_particle;
    }

    p1=ion[i];

    flag=0;
    for(j=0;j<(N*Np)/THREAD_N;j++){

      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	p->x=gsl_ran_flat(rnd_p[i],2.*dx,(Grid_Nx+2.)*dx);
	p->z=gsl_ran_flat(rnd_p[i],0,dx);
	p->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	p->vz=V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	p->n=N0i;
	p->flag=1;

	p->z+=p->vz*dtf*250.;

	if(p->z<0.){
	  p->z=(Grid_Nz+2.)*dx+p->z;
	  ion[i]->next_particle=p;

	  flag++;
	  if(flag==Np*Np){
	    flag=0;
	    ion[i]->flag=1;
	  }

	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else if(p->z>dx){
	  p->z+=dx;
	  ion[i]->next_particle=p;

	  flag++;
	  if(flag==Np*Np){
	    flag=0;
	    ion[i]->flag=1;
	  }

	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  free(p);
	}

      }else{
	printf("Can't allocalte memory\n");
	exit(0);
      }
    }   
  }

  if(myrank==0){
#pragma omp parallel for private(p,p1,j)
    for(i=0;i<THREAD_N;i++){
      ion[i]=ion_start[i];
      while(ion[i]->next_particle!=NULL){
	ion[i]=ion[i]->next_particle;
      }
      
      p1=ion[i];

      flag=0;
      for(j=0;j<(Np*Step_p)/THREAD_N;j++){
	
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],0.,dx);
	  p->z=gsl_ran_flat(rnd_p[i],2.*dx,(Grid_Nz+2.)*dx);
	  p->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	  p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	  p->vz=V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	  p->n=N0i;
	  p->flag=1;

	  p->x+=p->vx*dtf*250.;
	  
	  if(p->x>dx){
	    p->x+=dx;
	    ion[i]->next_particle=p;

	    flag++;
	    if(flag==Np*Np){
	      flag=0;
	      ion[i]->flag=1;
	    }

	    ion[i]->next_particle->prev_particle=ion[i];
	    ion[i]=ion[i]->next_particle;
	    ion[i]->next_particle=NULL;
	  }else{
	    free(p);
	  }
	  
	}else{
	  printf("Can't allocalte memory\n");
	  exit(0);
	}
      }   
    }
  }

  if(myrank==mpi-1){
#pragma omp parallel for private(p,p1,j)
    for(i=0;i<THREAD_N;i++){
      ion[i]=ion_start[i];
      while(ion[i]->next_particle!=NULL){
	ion[i]=ion[i]->next_particle;
      }
      
      p1=ion[i];

      flag=0;
      for(j=0;j<(Np*Step_p)/THREAD_N;j++){
	
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],0.,dx);
	  p->z=gsl_ran_flat(rnd_p[i],2.*dx,(Grid_Nz+2.)*dx);
	  p->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	  p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	  p->vz=V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	  p->n=N0i;
	  p->flag=1;
	  
	  p->x+=p->vx*dtf*250.;
	  
	  if(p->x<0.){
	    p->x=(Grid_Nx+2.)*dx+p->x;
	    ion[i]->next_particle=p;

	    flag++;
	    if(flag==Np*Np){
	      flag=0;
	      ion[i]->flag=1;
	    }

	    ion[i]->next_particle->prev_particle=ion[i];
	    ion[i]=ion[i]->next_particle;
	    ion[i]->next_particle=NULL;
	  }else{
	    free(p);
	  }
	  
	}else{
	  printf("Can't allocalte memory\n");
	  exit(0);
	}
      }   
    }
  }

  return(0);
}

int add_electron3(Particle *electron[],Particle *electron_start[],gsl_rng *rnd_p[],gsl_rng *rnd_v[],const int myrank,const int mpi)
{
  int i,j;
  int flag;
  Particle *p,*p1;

#pragma omp parallel for private(p,p1,j)
  for(i=0;i<THREAD_N;i++){
    electron[i]=electron_start[i];
    while(electron[i]->next_particle!=NULL){
      electron[i]=electron[i]->next_particle;
    }

    p1=electron[i];

    flag=0;
    for(j=0;j<(N*Np)/THREAD_N;j++){
      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	p->x=gsl_ran_flat(rnd_p[i],2.*dx,(Grid_Nx+2.)*dx);
	p->z=gsl_ran_flat(rnd_p[i],0.,dx);
	p->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	p->vz=V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	p->n=N0e;
	p->flag=1;

	p->z+=p->vz*dtf*250.;

	if(p->z<0.){
	  p->z=(Grid_Nz+2.)*dx+p->z;
	  electron[i]->next_particle=p;

	  flag++;
	  if(flag==Np*Np){
	    flag=0;
	    electron[i]->flag=1;
	  }

	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else if(p->z>dx){
	  p->z+=dx;
	  electron[i]->next_particle=p;

	  flag++;
	  if(flag==Np*Np){
	    flag=0;
	    electron[i]->flag=1;
	  }

	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  free(p);
	}

      }else{
	printf("Can't allocalte memory\n");
	exit(0);
      }
    }   
  }

  if(myrank==0){
#pragma omp parallel for private(p,p1,j)
    for(i=0;i<THREAD_N;i++){
      electron[i]=electron_start[i];
      while(electron[i]->next_particle!=NULL){
	electron[i]=electron[i]->next_particle;
      }
      
      p1=electron[i];

      flag=0;
      for(j=0;j<(Np*Step_p)/THREAD_N;j++){
	
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],0.,dx);
	  p->z=gsl_ran_flat(rnd_p[i],2.*dx,(Grid_Nz+2.)*dx);
	  p->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	  p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	  p->vz=V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	  p->n=N0e;
	  p->flag=1;
	  
	  p->x+=p->vx*dtf*250.;
	  
	  if(p->x>dx){	    
	    p->x+=dx;
	    electron[i]->next_particle=p;

	    flag++;
	    if(flag==Np*Np){
	      flag=0;
	      electron[i]->flag=1;
	    }

	    electron[i]->next_particle->prev_particle=electron[i];
	    electron[i]=electron[i]->next_particle;
	    electron[i]->next_particle=NULL;
	  }else{
	    free(p);
	  }
	  
	}else{
	  printf("Can't allocalte memory\n");
	  exit(0);
	}
      }   
    }
  }

  if(myrank==mpi-1){
#pragma omp parallel for private(p,p1,j)
    for(i=0;i<THREAD_N;i++){
      electron[i]=electron_start[i];
      while(electron[i]->next_particle!=NULL){
	electron[i]=electron[i]->next_particle;
      }
      
      p1=electron[i];

      flag=0;
      for(j=0;j<(Np*Step_p)/THREAD_N;j++){
	
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],0.,dx);
	  p->z=gsl_ran_flat(rnd_p[i],2.*dx,(Grid_Nz+2.)*dx);
	  p->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	  p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	  p->vz=V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	  p->n=N0e;
	  p->flag=1;
	  
	  p->x+=p->vx*dtf*250.;
	  
	  if(p->x<0.){    
	    p->x=(Grid_Nx+2.)*dx+p->x;
	    electron[i]->next_particle=p;

	    flag++;
	    if(flag==Np*Np){
	      flag=0;
	      electron[i]->flag=1;
	    }

	    electron[i]->next_particle->prev_particle=electron[i];
	    electron[i]=electron[i]->next_particle;
	    electron[i]->next_particle=NULL;
	  }else{
	    free(p);
	  }
	  
	}else{
	  printf("Can't allocalte memory\n");
	  exit(0);
	}
      }   
    }
  }

  //fprintf(stderr,"in %d\n",count[0]);

  return(0);
}

/*******************************************************************
shape function
 grid->particle
******************************************************************/

inline int shape_func_ex_bz3(const Particle *p,double *ex,double *bz,const Grid grid[][Grid_Nz+4])
{
  double w[4];
  const int k=(int)(p->x/dx);
  const int m=(int)(p->z/dx-0.5);
  const int kkk=k+1;
  const int mmm=m+1;

  w[0]=((k+1)*dx-p->x)*((m+1.5)*dx-p->z);
  w[1]=(p->x-k*dx)*((m+1.5)*dx-p->z);
  w[2]=((k+1)*dx-p->x)*(p->z-(m+0.5)*dx);
  w[3]=(p->x-k*dx)*(p->z-(m+0.5)*dx);

  *ex=(grid[k][m].ex*w[0]+grid[kkk][m].ex*w[1]+grid[k][mmm].ex*w[2]+grid[kkk][mmm].ex*w[3])/(dx*dx);
  *bz=(grid[k][m].bz*w[0]+grid[kkk][m].bz*w[1]+grid[k][mmm].bz*w[2]+grid[kkk][mmm].bz*w[3])/(dx*dx);

  return(0);
}

inline int shape_func_ez_bx3(const Particle *p,double *ez,double *bx,const Grid grid[][Grid_Nz+4])
{
  double w[4];
  const int k=(int)(p->x/dx-0.5);
  const int m=(int)(p->z/dx);
  const int kkk=k+1;
  const int mmm=m+1;

  w[0]=((k+1.5)*dx-p->x)*((m+1)*dx-p->z);
  w[1]=(p->x-(k+0.5)*dx)*((m+1)*dx-p->z);
  w[2]=((k+1.5)*dx-p->x)*(p->z-m*dx);
  w[3]=(p->x-(k+0.5)*dx)*(p->z-m*dx);

  *ez=(grid[k][m].ez*w[0]+grid[kkk][m].ez*w[1]+grid[k][mmm].ez*w[2]+grid[kkk][mmm].ez*w[3])/(dx*dx);
  *bx=(grid[k][m].bx*w[0]+grid[kkk][m].bx*w[1]+grid[k][mmm].bx*w[2]+grid[kkk][mmm].bx*w[3])/(dx*dx);

  return(0);
}

inline int shape_func_ey3(const Particle *p,double *ey,const Grid grid[][Grid_Nz+4])
{
  const int k=(int)(p->x/dx-0.5);
  const int m=(int)(p->z/dx-0.5);
  const int kkk=k+1;
  const int mmm=m+1;

  *ey=(grid[k][m].ey*((k+1.5)*dx-p->x)*((m+1.5)*dx-p->z)+grid[kkk][m].ey*(p->x-(k+0.5)*dx)*((m+1.5)*dx-p->z)
       +grid[k][mmm].ey*((k+1.5)*dx-p->x)*(p->z-(m+0.5)*dx)+grid[kkk][mmm].ey*(p->x-(k+0.5)*dx)*(p->z-(m+0.5)*dx))/(dx*dx);

  return(0);
}


inline int shape_func_by3(const Particle *p,double *by,const Grid grid[][Grid_Nz+4])
{
  const int k=(int)(p->x/dx);
  const int m=(int)(p->z/dx);
  const int kkk=k+1;
  const int mmm=m+1;

  *by=(grid[k][m].by*((k+1)*dx-p->x)*((m+1)*dx-p->z)+grid[kkk][m].by*(p->x-(k)*dx)*((m+1)*dx-p->z)
       +grid[k][mmm].by*((k+1)*dx-p->x)*(p->z-(m)*dx)+grid[kkk][mmm].by*(p->x-(k)*dx)*(p->z-(m)*dx))/(dx*dx);

  return(0);
}

/*******************************************************************
shape function
 particle->grid
******************************************************************/

inline int shape_func_ion0_jy(const Particle *p,const int thread,Grid_thread grid_thread[][Grid_Nx+4][Grid_Nz+4])
{
  const int k=(int)(p->x/dx-0.5);
  const int m=(int)(p->z/dx-0.5);
  const int kkk=k+1;
  const int mmm=m+1;
  const double temp=q*p->n*p->vy/(dx*dx);

  grid_thread[thread][k][m].jiy0+=temp*((k+1.5)*dx-p->x)*((m+1.5)*dx-p->z);
  grid_thread[thread][kkk][m].jiy0+=temp*(p->x-(k+0.5)*dx)*((m+1.5)*dx-p->z);
  grid_thread[thread][k][mmm].jiy0+=temp*((k+1.5)*dx-p->x)*(p->z-(m+0.5)*dx);
  grid_thread[thread][kkk][mmm].jiy0+=temp*(p->x-(k+0.5)*dx)*(p->z-(m+0.5)*dx);

  return(0);
}

inline int shape_func_ion0_n(const Particle *p,const int thread,Grid_thread grid_thread[][Grid_Nx+4][Grid_Nz+4])
{
  const int k=(int)(p->x/dx-0.5);
  const int m=(int)(p->z/dx-0.5);
  const int kkk=k+1;
  const int mmm=m+1;
  const double temp=p->n/(dx*dx);

  grid_thread[thread][k][m].ni+=temp*((k+1.5)*dx-p->x)*((m+1.5)*dx-p->z);
  grid_thread[thread][kkk][m].ni+=temp*(p->x-(k+0.5)*dx)*((m+1.5)*dx-p->z);
  grid_thread[thread][k][mmm].ni+=temp*((k+1.5)*dx-p->x)*(p->z-(m+0.5)*dx);
  grid_thread[thread][kkk][mmm].ni+=temp*(p->x-(k+0.5)*dx)*(p->z-(m+0.5)*dx);

  return(0);
}

inline int shape_func_ion0_jz(const Particle *p,const int thread,Grid_thread grid_thread[][Grid_Nx+4][Grid_Nz+4])
{
  const int k=(int)(p->x/dx-0.5);
  const int m=(int)(p->z/dx);
  const int kkk=k+1;
  const int mmm=m+1;
  const double temp=q*p->n*p->vz/(dx*dx);

  grid_thread[thread][k][m].jiz0+=temp*((k+1.5)*dx-p->x)*((m+1)*dx-p->z);
  grid_thread[thread][kkk][m].jiz0+=temp*(p->x-(k+0.5)*dx)*((m+1)*dx-p->z);
  grid_thread[thread][k][mmm].jiz0+=temp*((k+1.5)*dx-p->x)*(p->z-(m)*dx);
  grid_thread[thread][kkk][mmm].jiz0+=temp*(p->x-(k+0.5)*dx)*(p->z-(m)*dx);

  return(0);
}

inline int shape_func_ion0_jx(const Particle *p,const int thread,Grid_thread grid_thread[][Grid_Nx+4][Grid_Nz+4])
{
  const int k=(int)(p->x/dx);
  const int m=(int)(p->z/dx-0.5);
  const int kkk=k+1;
  const int mmm=m+1;
  const double temp=q*p->n*p->vx/(dx*dx);

  grid_thread[thread][k][m].jix0+=temp*((k+1)*dx-p->x)*((m+1.5)*dx-p->z);
  grid_thread[thread][kkk][m].jix0+=temp*(p->x-(k)*dx)*((m+1.5)*dx-p->z);
  grid_thread[thread][k][mmm].jix0+=temp*((k+1)*dx-p->x)*(p->z-(m+0.5)*dx);
  grid_thread[thread][kkk][mmm].jix0+=temp*(p->x-(k)*dx)*(p->z-(m+0.5)*dx);

  return(0);
}

inline int shape_func_electron0_jy(const Particle *p,const int thread,Grid_thread grid_thread[][Grid_Nx+4][Grid_Nz+4])
{
  const int k=(int)(p->x/dx-0.5);
  const int m=(int)(p->z/dx-0.5);
  const int kkk=k+1;
  const int mmm=m+1;;
  const double temp=q*p->n*p->vy/(dx*dx);

  grid_thread[thread][k][m].jey0+=temp*((k+1.5)*dx-p->x)*((m+1.5)*dx-p->z);
  grid_thread[thread][kkk][m].jey0+=temp*(p->x-(k+0.5)*dx)*((m+1.5)*dx-p->z);
  grid_thread[thread][k][mmm].jey0+=temp*((k+1.5)*dx-p->x)*(p->z-(m+0.5)*dx);
  grid_thread[thread][kkk][mmm].jey0+=temp*(p->x-(k+0.5)*dx)*(p->z-(m+0.5)*dx);

  return(0);
}

inline int shape_func_electron0_n(const Particle *p,const int thread,Grid_thread grid_thread[][Grid_Nx+4][Grid_Nz+4])
{
  const int k=(int)(p->x/dx-0.5);
  const int m=(int)(p->z/dx-0.5);
  const int kkk=k+1;
  const int mmm=m+1;
  const double temp=p->n/(dx*dx);

  grid_thread[thread][k][m].ne+=temp*((k+1.5)*dx-p->x)*((m+1.5)*dx-p->z);
  grid_thread[thread][kkk][m].ne+=temp*(p->x-(k+0.5)*dx)*((m+1.5)*dx-p->z);
  grid_thread[thread][k][mmm].ne+=temp*((k+1.5)*dx-p->x)*(p->z-(m+0.5)*dx);
  grid_thread[thread][kkk][mmm].ne+=temp*(p->x-(k+0.5)*dx)*(p->z-(m+0.5)*dx);

  return(0);
}

inline int shape_func_electron0_jz(const Particle *p,const int thread,Grid_thread grid_thread[][Grid_Nx+4][Grid_Nz+4])
{
  const int k=(int)(p->x/dx-0.5);
  const int m=(int)(p->z/dx);
  const int kkk=k+1;
  const int mmm=m+1;
  const double temp=q*p->n*p->vz/(dx*dx);

  grid_thread[thread][k][m].jez0+=temp*((k+1.5)*dx-p->x)*((m+1)*dx-p->z);
  grid_thread[thread][kkk][m].jez0+=temp*(p->x-(k+0.5)*dx)*((m+1)*dx-p->z);
  grid_thread[thread][k][mmm].jez0+=temp*((k+1.5)*dx-p->x)*(p->z-(m)*dx);
  grid_thread[thread][kkk][mmm].jez0+=temp*(p->x-(k+0.5)*dx)*(p->z-(m)*dx);

  return(0);
}

inline int shape_func_electron0_jx(const Particle *p,const int thread,Grid_thread grid_thread[][Grid_Nx+4][Grid_Nz+4])
{
  const int k=(int)(p->x/dx);
  const int m=(int)(p->z/dx-0.5);
  const int kkk=k+1;
  const int mmm=m+1;
  const double temp=q*p->n*p->vx/(dx*dx);

  grid_thread[thread][k][m].jex0+=temp*((k+1)*dx-p->x)*((m+1.5)*dx-p->z);
  grid_thread[thread][kkk][m].jex0+=temp*(p->x-(k)*dx)*((m+1.5)*dx-p->z);
  grid_thread[thread][k][mmm].jex0+=temp*((k+1)*dx-p->x)*(p->z-(m+0.5)*dx);
  grid_thread[thread][kkk][mmm].jex0+=temp*(p->x-(k)*dx)*(p->z-(m+0.5)*dx);

  return(0);
}

int output(Grid grid[][Grid_Nz+4],const int c,const int myrank,const int p)
{
  int i,j,t;
  int k,m;
  FILE *fp1;
  char filename[256];

  sprintf(filename,"pic%d-%d-%d-all.txt",version,c,myrank);
  fp1=fopen(filename,"w");

  for(k=2;k<Grid_Nx+2;k+=1){
    for(m=2;m<Grid_Nz+2;m+=1){
      fprintf(fp1,"%.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E\n",
	      k*dx-2.*dx+myrank*dx*Grid_Nx-p/2*Grid_Nx*dx,
	      m*dx-2.*dx-Grid_Nz/2*dx,
	      (grid[k-1][m-1].ni+grid[k-1][m].ni+grid[k][m-1].ni+grid[k][m].ni)/4.,
	      (grid[k-1][m-1].ne+grid[k-1][m].ne+grid[k][m-1].ne+grid[k][m].ne)/4.,
	      (grid[k-1][m-1].ni+grid[k-1][m].ni+grid[k][m-1].ni+grid[k][m].ni)/4.
	      -(grid[k-1][m-1].ne+grid[k-1][m].ne+grid[k][m-1].ne+grid[k][m].ne)/4.,
	      (grid[k][m-1].ex+grid[k][m].ex)/2.+V*IMF_y,
	      (grid[k-1][m-1].ey+grid[k-1][m].ey+grid[k][m-1].ey+grid[k][m].ey)/4.-V*IMF_x,
	      (grid[k-1][m].ez+grid[k][m].ez)/2.,
	      (grid[k-1][m].bx+grid[k][m].bx)/2.,
	      grid[k][m].by,
	      (grid[k][m-1].bz+grid[k][m].bz)/2.,
	      (grid[k][m-1].jix0+grid[k][m].jix0)/2.,
	      (grid[k-1][m-1].jiy0+grid[k-1][m].jiy0+grid[k][m-1].jiy0+grid[k][m].jiy0)/4.,
	      (grid[k-1][m].jiz0+grid[k][m].jiz0)/2.,
	      (grid[k][m-1].jex0+grid[k][m].jex0)/2.,
	      (grid[k-1][m-1].jey0+grid[k-1][m].jey0+grid[k][m-1].jey0+grid[k][m].jey0)/4.,
	      (grid[k-1][m].jez0+grid[k][m].jez0)/2.,
	      (grid[k][m-1].jix0+grid[k][m].jix0)/2.-(grid[k][m-1].jex0+grid[k][m].jex0)/2.,
	      (grid[k-1][m-1].jiy0+grid[k-1][m].jiy0+grid[k][m-1].jiy0+grid[k][m].jiy0)/4.
	      -(grid[k-1][m-1].jey0+grid[k-1][m].jey0+grid[k][m-1].jey0+grid[k][m].jey0)/4.,
	      (grid[k-1][m].jiz0+grid[k][m].jiz0)/2.-(grid[k-1][m].jez0+grid[k][m].jez0)/2.,
	      mi/q*(grid[k][m-1].jix0+grid[k][m].jix0)/2.+me/q*(grid[k][m-1].jex0+grid[k][m].jex0)/2.,
	      mi/q*(grid[k-1][m-1].jiy0+grid[k-1][m].jiy0+grid[k][m-1].jiy0+grid[k][m].jiy0)/4.
	      +me/q*(grid[k-1][m-1].jey0+grid[k-1][m].jey0+grid[k][m-1].jey0+grid[k][m].jey0)/4.,
	      mi/q*(grid[k-1][m].jiz0+grid[k][m].jiz0)/2.+me/q*(grid[k-1][m].jez0+grid[k][m].jez0)/2.,
	      grid[k][m].phi
	      );
      //x z ni ne dens ex ey ez bx by bz jix jiy jiz jex jey jez jx jy jz mvx mvy mvz
      //1 2 3  4  5    6  7  8  9  10 11 12  13  14  15  16  17  18 19 20 21  22  23
    }
    fprintf(fp1,"\n");
  }

  fclose(fp1);

  return(0);
}

int output_p(Particle *ion[],Particle *ion_start[],Particle *electron[],Particle *electron_start[],const int c,const int myrank,const int p)
{
  int i;
  char filename[256];
  FILE *fp_i,*fp_e;

  if(myrank==p/2-2||myrank==p/2-1||myrank==p/2||myrank==p/2+1){
    sprintf(filename,"pic%d-%d-%d-ion.txt",version,c,myrank);
    fp_i=fopen(filename,"w");

    sprintf(filename,"pic%d-%d-%d-electron.txt",version,c,myrank);
    fp_e=fopen(filename,"w");

    for(i=0;i<THREAD_N;i++){
      ion[i]=ion_start[i];
      while(ion[i]->next_particle!=NULL){
	if(ion[i]->flag==1){
	  fprintf(fp_i,"%.3E %.3E %.3E %.3E\n",ion[i]->x-2.*dx+myrank*dx*Grid_Nx-p/2*Grid_Nx*dx,ion[i]->z-Grid_Nz*dx/2,ion[i]->vx,ion[i]->vz);
	}
	ion[i]=ion[i]->next_particle;
      }
    }

    for(i=0;i<THREAD_N;i++){
      electron[i]=electron_start[i];
      while(electron[i]->next_particle!=NULL){
	if(electron[i]->flag==1){
	  fprintf(fp_e,"%.3E %.3E %.3E %.3E\n",electron[i]->x-2.*dx+myrank*dx*Grid_Nx-p/2*Grid_Nx*dx,electron[i]->z-Grid_Nz*dx/2,electron[i]->vx,electron[i]->vz);
	}
	electron[i]=electron[i]->next_particle;
      }
    }

    fclose(fp_i);
    fclose(fp_e);
  }

  return(0);
}

int external_current_perp(Grid grid[][Grid_Nz+4],const int c,const int myrank,const int p)
{
  int i,k,m;
  const double w=Pi/10000.;
  const double w2=Pi/500.;

  if(c<10000){
    if(myrank==p/2){
      k=3;
      m=(Grid_Nz+4)/2-3;
      grid[k][m].jiy0=-I/dx/dx*(1.+cos(w*c-Pi))/2.;

      k=3;
      m=(Grid_Nz+4)/2+2;
      grid[k][m].jiy0=I/dx/dx*(1.+cos(w*c-Pi))/2.;
    }
  }else{
    if(myrank==p/2){
      k=3;
      m=(Grid_Nz+4)/2-3;
      grid[k][m].jiy0=-I/dx/dx;

      k=3;
      m=(Grid_Nz+4)/2+2;
      grid[k][m].jiy0=I/dx/dx;
    }
  }

  return(0);
}

int external_current_para(Grid grid[][Grid_Nz+4],const int c,const int myrank,const int p)
{
  int i,k,m;
  const double w=Pi/10000.;
  const double w2=Pi/500.;

  if(c<10000){
    if(myrank==p/2-1){
      k=Grid_Nx-1;
      m=(Grid_Nz+4)/2-1;
      grid[k][m].jiy0=I/dx/dx*(1.+cos(w*c-Pi))/2.;
    }
    
    if(myrank==p/2){
      k=4;
      m=(Grid_Nz+4)/2-1;
      grid[k][m].jiy0=-I/dx/dx*(1.+cos(w*c-Pi))/2.;
    }
  }else{
    if(myrank==p/2-1){
      k=Grid_Nx-1;
      m=(Grid_Nz+4)/2-1;
      grid[k][m].jiy0=I/dx/dx;
    }
    
    if(myrank==p/2){
      k=4;
      m=(Grid_Nz+4)/2-1;
      grid[k][m].jiy0=-I/dx/dx;
    }
  }
  
  return(0);
}

int thrust_F(const Grid grid[][Grid_Nz+4],const int myrank,const int p,double *Fx,double *Fz,double *My)
{
  int i;
  int k,m;
  int dest,src,tag=1000;
  double x,z;
  double fx0,fz0,fx1,fz1;
  double F_r[PROCESS_N][3];
  MPI_Status stat[PROCESS_N];
  MPI_Request request[PROCESS_N];
  char filename[256];
  FILE *fp1;

  /*double xf0=2.5*dx;
  double zf0=-0.5*dx;
  double xb0=-2.5*dx;
  double zb0=-0.5*dx;*/

  double xf0=1.5*dx;
  double zf0=-2.5*dx;
  double xb0=1.5*dx;
  double zb0=2.5*dx;

  *Fx=*Fz=*My=0.;

  for(k=2;k<Grid_Nx+2;k++){
    for(m=2;m<Grid_Nz+2;m++){
      x=(myrank*Grid_Nx+(k-2)+0.5)*dx-dx*p*Grid_Nx/2.;
      z=(m-2)*dx-Grid_Nz/2*dx+0.5*dx;

      if(x!=xf0&&z!=zf0){
	fx0=-mu0*I*dx*dx*(grid[k][m].jiy0-grid[k][m].jey0)*(x-xf0)
	  /(2.*Pi*(pow(x-xf0,2)+pow(z-zf0,2)));
	fz0=-mu0*I*dx*dx*(grid[k][m].jiy0-grid[k][m].jey0)*(z-zf0)
	  /(2.*Pi*(pow(x-xf0,2)+pow(z-zf0,2))); 
      }

      if(x!=xb0&&z!=zb0){
	fx1=mu0*I*dx*dx*(grid[k][m].jiy0-grid[k][m].jey0)*(x-xb0)
	  /(2.*Pi*(pow(x-xb0,2)+pow(z-zb0,2)));
	fz1=mu0*I*dx*dx*(grid[k][m].jiy0-grid[k][m].jey0)*(z-zb0)
	  /(2.*Pi*(pow(x-xb0,2)+pow(z-zb0,2)));
      }

      *Fx+=fx0+fx1;

      *Fz+=fz0+fz1;

      *My+=(-(zf0+0.5*dx)*fx0+xf0*fz0)+(-(zb0+0.5*dx)*fx1+xb0*fz1);
    }
  }

  //printf("%E %E %E\n",*Fx,*Fz,*My);
  if(myrank==0){
    for(i=1;i<p;i++){
      src=i;
      MPI_Irecv(&F_r[i],3,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request[i]);
    }

    for(i=1;i<p;i++){
      MPI_Wait(&request[i],&stat[i]);
      
      *Fx+=F_r[i][0];
      *Fz+=F_r[i][1];
      *My+=F_r[i][2];
    }
  }else{
    F_r[myrank][0]=*Fx;
    F_r[myrank][1]=*Fz;
    F_r[myrank][2]=*My;

    dest=0;
    MPI_Send(&F_r[myrank],3,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);
  }

  if(myrank==0){
    sprintf(filename,"pic%d-f.txt",version);
    fp1=fopen(filename,"a");
    fprintf(fp1,"%E %E %E\n",*Fx,*Fz,*My);
    fclose(fp1);
  }

  return(0);
}

int thrust_F2(const Grid grid[][Grid_Nz+4],const int myrank,const int p,double *Fx2,double *Fz2)
{
  int i;
  int k,m;
  int dest,src,tag=1000;
  double F_r[PROCESS_N][2];
  MPI_Status stat[PROCESS_N];
  MPI_Request request[PROCESS_N];
  char filename[256];
  FILE *fp1;
  double bs,bx,by,bz;
  
  *Fx2=*Fz2=0.;
  
  for(k=2;k<Grid_Nx+2;k++){
    for(m=2;m<Grid_Nz+2;m++){
      if(grid[k][m].ni>10.){
	if(k-2+myrank*Grid_Nx-p/2*Grid_Nx==-p*Grid_Nx/4){
	  bx=(grid[k][m].bx+grid[k][m+1].bx)/2.;
	  by=(grid[k][m].by+grid[k+1][m].by+grid[k][m+1].by+grid[k+1][m+1].by)/4.;
	  bz=(grid[k][m].bz+grid[k+1][m].bz)/2.;
	  
	  bs=pow(bx,2)+pow(by,2)+pow(bz,2);
	  
	  *Fx2+=(bs/2./mu0-pow(bx,2)/mu0)*dx;
	  *Fz2+=(-bx*bz/mu0)*dx;
	}else if(k-2+myrank*Grid_Nx-p/2*Grid_Nx==p*Grid_Nx/4){ 
	  bx=(grid[k][m].bx+grid[k][m+1].bx)/2.;
	  by=(grid[k][m].by+grid[k+1][m].by+grid[k][m+1].by+grid[k+1][m+1].by)/4.;
	  bz=(grid[k][m].bz+grid[k+1][m].bz)/2.;
	  
	  bs=pow(bx,2)+pow(by,2)+pow(bz,2);
	  
	  *Fx2-=(+bs/2./mu0-pow(bx,2)/mu0)*dx;
	  *Fz2-=(-bx*bz/mu0)*dx;
	}else if(m-2-Grid_Nz/2==-Grid_Nz/4){	  
	  bx=(grid[k][m].bx+grid[k][m+1].bx)/2.;
	  by=(grid[k][m].by+grid[k+1][m].by+grid[k][m+1].by+grid[k+1][m+1].by)/4.;
	  bz=(grid[k][m].bz+grid[k+1][m].bz)/2.;
	  
	  bs=pow(bx,2)+pow(by,2)+pow(bz,2);
	  
	  *Fx2+=(-bx*bz/mu0)*dx;
	  *Fz2+=(+bs/2./mu0-pow(bz,2)/mu0)*dx;
	}else if(m-2-Grid_Nz/2==Grid_Nz/4){
	  bx=(grid[k][m].bx+grid[k][m+1].bx)/2.;
	  by=(grid[k][m].by+grid[k+1][m].by+grid[k][m+1].by+grid[k+1][m+1].by)/4.;
	  bz=(grid[k][m].bz+grid[k+1][m].bz)/2.;
	  
	  bs=pow(bx,2)+pow(by,2)+pow(bz,2);
	  
	  *Fx2-=(-bx*bz/mu0)*dx;
	  *Fz2-=(+bs/2./mu0-pow(bz,2)/mu0)*dx;
	}
      }
    }
  }

  if(myrank==0){
    for(i=1;i<p;i++){
      src=i;
      MPI_Irecv(&F_r[i],2,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request[i]);
    }

    for(i=1;i<p;i++){
      MPI_Wait(&request[i],&stat[i]);
      
      *Fx2+=F_r[i][0];
      *Fz2+=F_r[i][1];
    }
  }else{
    F_r[myrank][0]=*Fx2;
    F_r[myrank][1]=*Fz2;

    dest=0;
    MPI_Send(&F_r[myrank],2,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);
  }

  if(myrank==0){
    sprintf(filename,"pic%d-f2.txt",version);
    fp1=fopen(filename,"a");
    fprintf(fp1,"%E %E\n",-*Fx2,-*Fz2);
    fclose(fp1);
  }

  return(0);
}

int thrust_F3(const Grid grid[][Grid_Nz+4],const int myrank,const int p,double *Fx3,double *Fz3)
{
  int i;
  int k,m;
  int dest,src,tag=1000;
  double F_r[PROCESS_N][2];
  MPI_Status stat[PROCESS_N];
  MPI_Request request[PROCESS_N];
  char filename[256];
  FILE *fp1;
  double bs,bx,by,bz;

  *Fx3=*Fz3=0.;

  for(k=2;k<Grid_Nx+2;k++){
    for(m=2;m<Grid_Nz+2;m++){
      if(grid[k][m].ni>10.){
	if(k-2+myrank*Grid_Nx-p/2*Grid_Nx==-p*Grid_Nx/4){
	  bx=(grid[k][m].bx+grid[k][m+1].bx)/2.;
	  by=(grid[k][m].by+grid[k+1][m].by+grid[k][m+1].by+grid[k+1][m+1].by)/4.;
	  bz=(grid[k][m].bz+grid[k+1][m].bz)/2.;
	  
	  bs=pow(bx,2)+pow(by,2)+pow(bz,2);
	  
	  *Fx3+=(mi*grid[k][m].ni
		 *pow((grid[k][m].jix0+grid[k][m+1].jix0)/2./q/grid[k][m].ni,2)
		 +bs/2./mu0
		 -pow(bx,2)/mu0)*dx;
	  *Fz3+=(mi*grid[k][m].ni
		 *((grid[k][m].jix0+grid[k][m+1].jix0)/2./q/grid[k][m].ni)
		 *((grid[k][m].jiz0+grid[k+1][m].jiz0)/2./q/grid[k][m].ni)
		 -bx*bz/mu0)*dx;
	}else if(k-2+myrank*Grid_Nx-p/2*Grid_Nx==p*Grid_Nx/4){
	  bx=(grid[k][m].bx+grid[k][m+1].bx)/2.;
	  by=(grid[k][m].by+grid[k+1][m].by+grid[k][m+1].by+grid[k+1][m+1].by)/4.;
	  bz=(grid[k][m].bz+grid[k+1][m].bz)/2.;
	  
	  bs=pow(bx,2)+pow(by,2)+pow(bz,2);
	  
	  *Fx3-=(mi*grid[k][m].ni
		 *pow((grid[k][m].jix0+grid[k][m+1].jix0)/2./q/grid[k][m].ni,2)
		 +bs/2./mu0
		 -pow(bx,2)/mu0)*dx;
	  *Fz3-=(mi*grid[k][m].ni
		 *((grid[k][m].jix0+grid[k][m+1].jix0)/2./q/grid[k][m].ni)
		 *((grid[k][m].jiz0+grid[k+1][m].jiz0)/2./q/grid[k][m].ni)
		 -bx*bz/mu0)*dx;
	}else if(m-2-Grid_Nz/2==-Grid_Nz/4){
	  bx=(grid[k][m].bx+grid[k][m+1].bx)/2.;
	  by=(grid[k][m].by+grid[k+1][m].by+grid[k][m+1].by+grid[k+1][m+1].by)/4.;
	  bz=(grid[k][m].bz+grid[k+1][m].bz)/2.;
	  
	  bs=pow(bx,2)+pow(by,2)+pow(bz,2);
	  
	  *Fx3+=(mi*grid[k][m].ni
		 *((grid[k][m].jix0+grid[k][m+1].jix0)/2./q/grid[k][m].ni)
		 *((grid[k][m].jiz0+grid[k+1][m].jiz0)/2./q/grid[k][m].ni)
		 -bx*bz/mu0)*dx;
	  *Fz3+=(mi*grid[k][m].ni
		 *pow((grid[k][m].jiz0+grid[k+1][m].jiz0)/2./q/grid[k][m].ni,2)
		 +bs/2./mu0
		 -pow(bz,2)/mu0)*dx;
	}else if(m-2-Grid_Nz/2==Grid_Nz/4){
	  bx=(grid[k][m].bx+grid[k][m+1].bx)/2.;
	  by=(grid[k][m].by+grid[k+1][m].by+grid[k][m+1].by+grid[k+1][m+1].by)/4.;
	  bz=(grid[k][m].bz+grid[k+1][m].bz)/2.;
	  
	  bs=pow(bx,2)+pow(by,2)+pow(bz,2);
	  
	  *Fx3-=(mi*grid[k][m].ni
		 *((grid[k][m].jix0+grid[k][m+1].jix0)/2./q/grid[k][m].ni)
		 *((grid[k][m].jiz0+grid[k+1][m].jiz0)/2./q/grid[k][m].ni)
		 -bx*bz/mu0)*dx;
	  *Fz3-=(mi*grid[k][m].ni
		 *pow((grid[k][m].jiz0+grid[k+1][m].jiz0)/2./q/grid[k][m].ni,2)
		 +bs/2./mu0
		 -pow(bz,2)/mu0)*dx;
	}
      }
    }
  }

  if(myrank==0){
    for(i=1;i<p;i++){
      src=i;
      MPI_Irecv(&F_r[i],2,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request[i]);
    }

    for(i=1;i<p;i++){
      MPI_Wait(&request[i],&stat[i]);
      
      *Fx3+=F_r[i][0];
      *Fz3+=F_r[i][1];
    }
  }else{
    F_r[myrank][0]=*Fx3;
    F_r[myrank][1]=*Fz3;

    dest=0;
    MPI_Send(&F_r[myrank],2,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);
  }

  if(myrank==0){
    sprintf(filename,"pic%d-f3.txt",version);
    fp1=fopen(filename,"a");
    fprintf(fp1,"%E %E\n",-*Fx3,-*Fz3);
    fclose(fp1);
  }

  return(0);
}

int renew_phi(Grid grid[][Grid_Nz+4],const int myrank,const int p,const int c)
{
  int i,k,m;
  double A;
  int src,dest,tag,count;
  MPI_Status stat[PROCESS_N];
  MPI_Request request[PROCESS_N];
  MPI_Status stat2[PROCESS_N];
  MPI_Request request2[PROCESS_N];
  static double rho[Grid_Nx][Grid_Nz];
  static double rho2[Grid_Nz/PROCESS_N][Grid_Nx*PROCESS_N];
  static double rho_all[Grid_Nx*PROCESS_N][Grid_Nz];
  static double rho_all2[Grid_Nz][Grid_Nx*PROCESS_N];
  double phi[Grid_Nz+4];
  FILE* fp;
  char filename[256];
  
  tag=1000;
  
#pragma omp parallel for private(m)
  for(k=0;k<Grid_Nx;k++){
    for(m=0;m<Grid_Nz;m++){
      rho[k][m]=0.;
    }
  }

  if(myrank==0){
#pragma omp parallel for private(m)
    for(k=Absorb_grid3;k<Grid_Nx+2;k++){
      for(m=Absorb_grid3;m<Grid_Nz+4-Absorb_grid3;m++){
	rho[k-2][m-2]=-(-q*(grid[k][m].ni-grid[k][m].ne)/e0
			+(grid[k+1][m].ex-grid[k][m].ex)/dx+(grid[k][m+1].ez-grid[k][m].ez)/dx);
      }
    }
  }else if(myrank==p-1){
#pragma omp parallel for private(m)
    for(k=2;k<Grid_Nx+4-Absorb_grid3;k++){
      for(m=Absorb_grid3;m<Grid_Nz+4-Absorb_grid3;m++){
	rho[k-2][m-2]=-(-q*(grid[k][m].ni-grid[k][m].ne)/e0
			+(grid[k+1][m].ex-grid[k][m].ex)/dx+(grid[k][m+1].ez-grid[k][m].ez)/dx);
      }
    }
  }else{
#pragma omp parallel for private(m)
    for(k=2;k<Grid_Nx+2;k++){
      for(m=Absorb_grid3;m<Grid_Nz+4-Absorb_grid3;m++){
	rho[k-2][m-2]=-(-q*(grid[k][m].ni-grid[k][m].ne)/e0
			+(grid[k+1][m].ex-grid[k][m].ex)/dx+(grid[k][m+1].ez-grid[k][m].ez)/dx);
      }
    }
  }

  fft_z(rho);
  
  if(myrank==0){
    memcpy(&rho_all[0],&rho[0],Grid_Nx*Grid_Nz*sizeof(double));
    
    for(i=1;i<p;i++){
      MPI_Irecv(&rho_all[i*Grid_Nx],Grid_Nx*Grid_Nz,MPI_DOUBLE,i,tag,MPI_COMM_WORLD,&request[i]);
    }
    for(i=1;i<p;i++){
      MPI_Wait(&request[i],&stat[i]);
    }

#pragma omp parallel for private(m)
    for(k=0;k<Grid_Nx*PROCESS_N;k++){
      for(m=0;m<Grid_Nz;m++){
	rho_all2[m][k]=rho_all[k][m];
      }
    }
    
    memcpy(&rho2[0],&rho_all2[0],Grid_Nx*Grid_Nz*sizeof(double));
    
    for(i=1;i<p;i++){
      MPI_Send(&rho_all2[i*Grid_Nz/PROCESS_N],Grid_Nx*Grid_Nz,MPI_DOUBLE,i,tag,MPI_COMM_WORLD);
    }
    
  }else{
    MPI_Send(&rho[0],Grid_Nx*Grid_Nz,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);
    
    MPI_Irecv(&rho2[0],Grid_Nx*Grid_Nz,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&request2[0]);
    
    MPI_Wait(&request2[0],&stat2[0]);
  }
  
  fft_x(rho2);
  
#pragma omp parallel for private(m,A)
  for(k=0;k<Grid_Nz/PROCESS_N;k++){
    for(m=0;m<Grid_Nx*PROCESS_N;m++){
      A=-(pow(sin(Pi*(double)(k+myrank*Grid_Nz/PROCESS_N)/(double)(Grid_Nz)/2.),2)
	  +pow(sin(Pi*(double)m/(double)(Grid_Nx*PROCESS_N)/2.),2))/pow(dx/2.,2);
      
      if(A==0.){
	rho2[k][m]=0.;
      }else{
	rho2[k][m]=rho2[k][m]/A;       
      } 
    }
  }
  
  inv_fft_x(rho2);
  
  if(myrank==0){
    memcpy(&rho_all2[0],&rho2[0],Grid_Nx*Grid_Nz*sizeof(double));
    
    for(i=1;i<p;i++){
      MPI_Irecv(&rho_all2[i*Grid_Nz/PROCESS_N],Grid_Nx*Grid_Nz,MPI_DOUBLE,i,tag,MPI_COMM_WORLD,&request2[i]);
    }
    for(i=1;i<p;i++){
      MPI_Wait(&request2[i],&stat2[i]);
    }

#pragma omp parallel for private(m)
    for(k=0;k<Grid_Nz;k++){
      for(m=0;m<Grid_Nx*PROCESS_N;m++){
	rho_all[m][k]=rho_all2[k][m];
      }
    }
    
    memcpy(&rho[0],&rho_all[0],Grid_Nx*Grid_Nz*sizeof(double));
    
    for(i=1;i<p;i++){
      MPI_Send(&rho_all[i*Grid_Nx],Grid_Nx*Grid_Nz,MPI_DOUBLE,i,tag,MPI_COMM_WORLD);
    }
    
  }else{
    MPI_Send(&rho2[0],Grid_Nx*Grid_Nz,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);
    
    MPI_Irecv(&rho[0],Grid_Nx*Grid_Nz,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&request[0]);
    
    MPI_Wait(&request[0],&stat[0]);
  }
  
  inv_fft_z(rho);

  MPI_Barrier(MPI_COMM_WORLD);
  
  count=Grid_Nz;
  
  src=myrank-1;
  
  if(src==-1){
    src=p-1;
  }
  
  MPI_Irecv(&phi,count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&request[0]);
  
  dest=myrank+1;
  
  if(dest==p){
    dest=0;
  }
  
  MPI_Send(&rho[Grid_Nx-1],count,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);
  
  MPI_Wait(&request[0],&stat[0]);
  
#pragma omp parallel for private(m)
  for(k=0;k<Grid_Nx;k++){
    for(m=0;m<Grid_Nz;m++){
      if(k==0){
	grid[k+2][m+2].ex+=(rho[k][m]-phi[m])/dx;
	grid[k+2][m+2].phi=rho[k][m];
      }else{
	grid[k+2][m+2].ex+=(rho[k][m]-rho[k-1][m])/dx;
	grid[k+2][m+2].phi=rho[k][m];
      }
      
      if(m==0){
	grid[k+2][m+2].ez+=(rho[k][m]-0.)/dx;
      }else{
	grid[k+2][m+2].ez+=(rho[k][m]-rho[k][m-1])/dx;
      }
    }
  }

  return(0);
}

/*******************************************************************
 fft for z-axis
 ******************************************************************/
inline int fft_z(double rho[][Grid_Nz])
{
  int k,m;
  int local_thread;
  static double data[THREAD_N][Grid_Nz+1];
  
#pragma omp parallel for private(m,local_thread)
  for(k=0;k<Grid_Nx;k++){
#ifdef _OPENMP
    local_thread=omp_get_thread_num();
#else
    local_thread=0;
#endif
    for(m=0;m<Grid_Nz;m++){
      data[local_thread][m+1]=rho[k][m];
    }
    
    sinft(Grid_Nz,data[local_thread]);
    
    for(m=0;m<Grid_Nz;m++){
      rho[k][m]=data[local_thread][m+1];
    }
  }
  
  return(0);
}

/*******************************************************************
 fft for x-axis
******************************************************************/
inline int fft_x(double rho2[][Grid_Nx*PROCESS_N])
{
  int k,m;
  int local_thread;
  static double data[THREAD_N][Grid_Nx*PROCESS_N+1];
  
#pragma omp parallel for private(m,local_thread)
  for(k=0;k<Grid_Nz/PROCESS_N;k++){
#ifdef _OPENMP
    local_thread=omp_get_thread_num();
#else
    local_thread=0;
#endif
    for(m=0;m<Grid_Nx*PROCESS_N;m++){
      data[local_thread][m+1]=rho2[k][m];
    }
    
    sinft(Grid_Nx*PROCESS_N,data[local_thread]);
    
    for(m=0;m<Grid_Nx*PROCESS_N;m++){
      rho2[k][m]=data[local_thread][m+1];
    }
  }
  
  return(0);
}

/*******************************************************************
 inverse fft for z-axis
******************************************************************/
inline int inv_fft_z(double rho[][Grid_Nz])
{
  int k,m;
  int local_thread;
  static double data[THREAD_N][Grid_Nz+1];
  
#pragma omp parallel for private(m,local_thread)
  for(k=0;k<Grid_Nx;k++){
#ifdef _OPENMP
    local_thread=omp_get_thread_num();
#else
    local_thread=0;
#endif
    for(m=0;m<Grid_Nz;m++){
      data[local_thread][m+1]=rho[k][m];
    }
    
    sinft(Grid_Nz,data[local_thread]);
    
    for(m=0;m<Grid_Nz;m++){
      rho[k][m]=2.*data[local_thread][m+1]/Grid_Nz;
    }
  }
  
  return(0);
}

/*******************************************************************
 inverse fft for x-axis
******************************************************************/
inline int inv_fft_x(double rho2[][Grid_Nx*PROCESS_N])
{
  int k,m;
  int local_thread;
  static double data[THREAD_N][Grid_Nx*PROCESS_N+1];

#pragma omp parallel for private(m,local_thread)
  for(k=0;k<Grid_Nz/PROCESS_N;k++){
#ifdef _OPENMP
    local_thread=omp_get_thread_num();
#else
    local_thread=0;
#endif
    for(m=0;m<Grid_Nx*PROCESS_N;m++){
      data[local_thread][m+1]=rho2[k][m];
    }
    
    sinft(Grid_Nx*PROCESS_N,data[local_thread]);
    
    for(m=0;m<Grid_Nx*PROCESS_N;m++){
      rho2[k][m]=2.*data[local_thread][m+1]/(Grid_Nx*PROCESS_N);
    }
  }
  
  return(0);
}

/*******************************************************************
fast sin fouir transform
******************************************************************/
inline int sinft(int n,double y[])
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
inline int realft(double data[],int n,int isign)
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
inline int four1(double data[],int nn,int isign)
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

int add_ion4(Particle *ion[],Particle *ion_start[],gsl_rng *rnd_p[],gsl_rng *rnd_v[],const int myrank,const int mpi)
{
  int i,j;
  int flag=0;
  Particle *p,*p1;
  const double pp=0.;
  const double pp2=2.;
  const double pz=3.;

  if(myrank==mpi/2-1){
#pragma omp parallel for private(p,p1,j)
    for(i=0;i<THREAD_N;i++){
      ion[i]=ion_start[i];
      while(ion[i]->next_particle!=NULL){
	ion[i]=ion[i]->next_particle;
      }
      
      p1=ion[i];
      for(j=0;j<(Np*Np)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],(Grid_Nx+1.-pp)*dx,(Grid_Nx+2.-pp)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],((Grid_Nz+4)/2-1-pz)*dx,((Grid_Nz+4)/2-pz)*dx);
	  p->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*10.*Ti/mi));
	  //p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	  p->vy=0.;
	  p->vz=-V/10.+gsl_ran_gaussian(rnd_v[i],sqrt(kb*10.*Ti/mi));
	  p->n=10.*N0i;
	  p->flag=3;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  printf("Can't allocalte memory\n");
	  exit(0);
	}
      }   
      for(j=0;j<(Np*Np)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],(Grid_Nx+1.-pp)*dx,(Grid_Nx+2.-pp)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],((Grid_Nz+4)/2+pz)*dx,((Grid_Nz+4)/2+1+pz)*dx);
	  p->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*10.*Ti/mi));
	  //p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	  p->vy=0.;
	  p->vz=V/10.+gsl_ran_gaussian(rnd_v[i],sqrt(kb*10.*Ti/mi));
	  p->n=10.*N0i;
	  p->flag=3;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  printf("Can't allocalte memory\n");
	  exit(0);
	  }
       }  
    }
  }else if(myrank==mpi/2){
#pragma omp parallel for private(p,p1,j)
    for(i=0;i<THREAD_N;i++){
      ion[i]=ion_start[i];
      while(ion[i]->next_particle!=NULL){
	ion[i]=ion[i]->next_particle;
      }
      
      p1=ion[i];
      for(j=0;j<(Np*Np)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],(2.+pp2)*dx,(3.+pp2)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],((Grid_Nz+4)/2-1-pz)*dx,((Grid_Nz+4)/2-pz)*dx);
	  p->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*10.*Ti/mi));
	  //p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	  p->vy=0.;
	  p->vz=-V/10.+gsl_ran_gaussian(rnd_v[i],sqrt(kb*10.*Ti/mi));
	  p->n=10.*N0i;
	  p->flag=3;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  printf("Can't allocalte memory\n");
	  exit(0);
	}
      }   
      for(j=0;j<(Np*Np)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],(2.+pp2)*dx,(3.+pp2)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],((Grid_Nz+4)/2+pz)*dx,((Grid_Nz+4)/2+1+pz)*dx);
	  p->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*10.*Ti/mi));
	  //p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	  p->vy=0.;
	  p->vz=V/10.+gsl_ran_gaussian(rnd_v[i],sqrt(kb*10.*Ti/mi));
	  p->n=10.*N0i;
	  p->flag=3;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  printf("Can't allocalte memory\n");
	  exit(0);
	}
      } 
    }
  }
  return(0);
}

int add_electron4(Particle *electron[],Particle *electron_start[],gsl_rng *rnd_p[],gsl_rng *rnd_v[],const int myrank,const int mpi)
{
  int i,j;
  Particle *p,*p1;
  const double pp=0.;
  const double pp2=2.;
  const double pz=5.;

  if(myrank==mpi/2-1){
#pragma omp parallel for private(p,p1,j)
    for(i=0;i<THREAD_N;i++){
      electron[i]=electron_start[i];
      while(electron[i]->next_particle!=NULL){
	electron[i]=electron[i]->next_particle;
      }
      
      p1=electron[i];
      for(j=0;j<(Np*Np)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],(Grid_Nx+1.-pp)*dx,(Grid_Nx+2.-pp)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],((Grid_Nz+4)/2-1-pz)*dx,((Grid_Nz+4)/2-pz)*dx);
	  p->vx=-50*V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*10*Te/me));
	  //p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*10*Te/me));
	  p->vy=0.;
	  p->vz=-50*V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*10*Te/me));
	  p->n=10.*N0e;
	  p->flag=3;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  printf("Can't allocalte memory\n");
	  exit(0);
	}
      }
      for(j=0;j<(Np*Np)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],(Grid_Nx+1.-pp)*dx,(Grid_Nx+2.-pp)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],((Grid_Nz+4)/2+pz)*dx,((Grid_Nz+4)/2+1+pz)*dx);
	  p->vx=-50*V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*10*Te/me));
	  //p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*10*Te/me));
	  p->vy=0.;
	  p->vz=50*V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*10*Te/me));
	  p->n=10.*N0e;
	  p->flag=3;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  printf("Can't allocalte memory\n");
	  exit(0);
	}
      }
    }
  }else if(myrank==mpi/2){
#pragma omp parallel for private(p,p1,j)
    for(i=0;i<THREAD_N;i++){
      electron[i]=electron_start[i];
      while(electron[i]->next_particle!=NULL){
	electron[i]=electron[i]->next_particle;
      }
      
      p1=electron[i];
      for(j=0;j<(Np*Np)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],(2.+pp2)*dx,(3.+pp2)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],((Grid_Nz+4)/2-1-pz)*dx,((Grid_Nz+4)/2-pz)*dx);
	  p->vx=50*V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*10*Te/me));
	  //p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*10*Te/me));
	  p->vy=0.;
	  p->vz=-50*V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*10*Te/me));
	  p->n=10.*N0e;
	  p->flag=3;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  printf("Can't allocalte memory\n");
	  exit(0);
	}
      }   
      for(j=0;j<(Np*Np)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],(2.+pp2)*dx,(3.+pp2)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],((Grid_Nz+4)/2+pz)*dx,((Grid_Nz+4)/2+1+pz)*dx);
	  p->vx=50*V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*10*Te/me));
	  //p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*10*Te/me));
	  p->vy=0.;
	  p->vz=50*V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*10*Te/me));
	  p->n=10.*N0e;
	  p->flag=3;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  printf("Can't allocalte memory\n");
	  exit(0);
	}
      } 
    }
  }

  return(0);
}

int sorting(Particle *ion[],Particle *ion_start[],Particle *electron[],Particle *electron_start[])
{
  Particle *sort_start[THREAD_N][Grid_Nx+4][Grid_Nz+4];
  Particle *sort[THREAD_N][Grid_Nx+4][Grid_Nz+4];
  int count[Grid_Nx+4][Grid_Nz+4];
  int i,j,k,m,flag;

  for(k=0;k<Grid_Nx+4;k++){
    for(m=0;m<Grid_Nz+4;m++){
      count[k][m]=0;
    }
  }

#pragma omp parallel for private(k,m,flag)  
  for(i=0;i<THREAD_N;i++){
    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){	
      if(count[(int)(ion[i]->x/dx)][(int)(ion[i]->z/dx)]==0){
	sort_start[i][(int)(ion[i]->x/dx)][(int)(ion[i]->z/dx)]=ion[i];
	sort_start[i][(int)(ion[i]->x/dx)][(int)(ion[i]->z/dx)]->prev_particle=NULL;
	sort[i][(int)(ion[i]->x/dx)][(int)(ion[i]->z/dx)]=sort_start[i][(int)(ion[i]->x/dx)][(int)(ion[i]->z/dx)];

	sort[i][(int)(ion[i]->x/dx)][(int)(ion[i]->z/dx)]->next_particle=NULL;

	count[(int)(ion[i]->x/dx)][(int)(ion[i]->z/dx)]=1;
      }else{
	sort[i][(int)(ion[i]->x/dx)][(int)(ion[i]->z/dx)]->next_particle=ion[i];
	sort[i][(int)(ion[i]->x/dx)][(int)(ion[i]->z/dx)]=sort[i][(int)(ion[i]->x/dx)][(int)(ion[i]->z/dx)]->next_particle->prev_particle=sort[i][(int)(ion[i]->x/dx)][(int)(ion[i]->z/dx)];
	sort[i][(int)(ion[i]->x/dx)][(int)(ion[i]->z/dx)]=sort[i][(int)(ion[i]->x/dx)][(int)(ion[i]->z/dx)]->next_particle;
	sort[i][(int)(ion[i]->x/dx)][(int)(ion[i]->z/dx)]->next_particle=NULL;
      }
      ion[i]=ion[i]->next_particle;
    }
    flag=0;
    for(k=0;k<Grid_Nx+4;k++){
      for(m=0;m<Grid_Nz+4;m++){
	if(count[k][m]!=0&&flag==0){
	  flag=1;
	  ion_start[i]=sort_start[i][k][m];
	  ion[i]=ion_start[i];
	  sort[i][k][m]=sort_start[i][k][m];

	  while(sort[i][k][m]->next_particle!=NULL){
	    ion[i]->next_particle=sort[i][k][m]->next_particle;	
	    ion[i]->prev_particle=sort[i][k][m]->prev_particle;	

	    ion[i]=ion[i]->next_particle;
	    sort[i][k][m]=sort[i][k][m]->next_particle;

	    ion[i]->next_particle=NULL;
	  }
	}else if(count[k][m]!=0&&flag!=0){
	}
      }
    }
  }

  return(0);
}

int add_ion5(Particle *ion[],Particle *ion_start[],gsl_rng *rnd_p[],gsl_rng *rnd_v[],const int myrank,const int mpi)
{
  int i,j;
  int flag=0;
  Particle *p;

#pragma omp parallel for  
  for(i=0;i<THREAD_N;i++){
    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){
      if(ion[i]->z<3*dx||ion[i]->z>(Grid_Nz+1)*dx
	 ||(myrank==0&&ion[i]->x<3*dx)||(myrank==mpi-1&&ion[i]->x>(Grid_Nx+1)*dx)){
	del_particle_i(ion,i,ion_start);
      }
      ion[i]=ion[i]->next_particle;
    }
    if(ion[i]->z<3*dx||ion[i]->z>(Grid_Nz+1)*dx
       ||(myrank==0&&ion[i]->x<3*dx)||(myrank==mpi-1&&ion[i]->x>(Grid_Nx+1)*dx)){
      del_particle_i(ion,i,ion_start);
    }
  }
  
#pragma omp parallel for private(p,j)
  for(i=0;i<THREAD_N;i++){
    ion[i]=ion_start[i];
    while(ion[i]->next_particle!=NULL){
      ion[i]=ion[i]->next_particle;
    }
    
    for(j=0;j<(N*Np)/THREAD_N;j++){
      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	p->x=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Nx+2)*dx);
	p->z=gsl_ran_flat(rnd_p[i],2*dx,3*dx);
	p->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	//p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	p->vy=0.;
	p->vz=V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	p->n=N0i;
	p->flag=1;
	  
	ion[i]->next_particle=p;
	ion[i]->next_particle->prev_particle=ion[i];
	ion[i]=ion[i]->next_particle;
	ion[i]->next_particle=NULL;
      }else{
	printf("Can't allocalte memory\n");
	exit(0);
      }
    }

    for(j=0;j<(N*Np)/THREAD_N;j++){
      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	p->x=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Nx+2)*dx);
	p->z=gsl_ran_flat(rnd_p[i],(Grid_Nz+1)*dx,(Grid_Nz+2)*dx);
	p->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	//p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	p->vy=0.;
	p->vz=V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	p->n=N0i;
	p->flag=1;
	  
	ion[i]->next_particle=p;
	ion[i]->next_particle->prev_particle=ion[i];
	ion[i]=ion[i]->next_particle;
	ion[i]->next_particle=NULL;
      }else{
	printf("Can't allocalte memory\n");
	exit(0);
      }
    }
  }

  if(myrank==0){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      
      for(j=0;j<((Step_p-2)*Np)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],2*dx,3*dx);
	  p->z=gsl_ran_flat(rnd_p[i],3*dx,(Grid_Nz+1)*dx);
	  p->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	  //p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	  p->vy=0.;
	  p->vz=V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	  p->n=N0i;
	  p->flag=1;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  printf("Can't allocalte memory\n");
	  exit(0);
	}
      }   
    }
  }

  if(myrank==mpi-1){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      
      for(j=0;j<((Step_p-2)*Np)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],(Grid_Nx+1)*dx,(Grid_Nx+2)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],3*dx,(Grid_Nz+1)*dx);
	  p->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	  //p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	  p->vy=0.;
	  p->vz=V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*Ti/mi));
	  p->n=N0i;
	  p->flag=1;
	  
	  ion[i]->next_particle=p;
	  ion[i]->next_particle->prev_particle=ion[i];
	  ion[i]=ion[i]->next_particle;
	  ion[i]->next_particle=NULL;
	}else{
	  printf("Can't allocalte memory\n");
	  exit(0);
	}
      }   
    }
  }
  return(0);
}

int add_electron5(Particle *electron[],Particle *electron_start[],gsl_rng *rnd_p[],gsl_rng *rnd_v[],const int myrank,const int mpi)
{
  int i,j;
  int flag=0;
  Particle *p;

#pragma omp parallel for  
  for(i=0;i<THREAD_N;i++){
    electron[i]=electron_start[i];
    while(electron[i]->next_particle!=NULL){
      if(electron[i]->z<3*dx||electron[i]->z>(Grid_Nz+1)*dx
	 ||(myrank==0&&electron[i]->x<3*dx)||(myrank==mpi-1&&electron[i]->x>(Grid_Nx+1)*dx)){
	del_particle_e(electron,i,electron_start);
      }
      electron[i]=electron[i]->next_particle;
    }
    if(electron[i]->z<3*dx||electron[i]->z>(Grid_Nz+1)*dx
       ||(myrank==0&&electron[i]->x<3*dx)||(myrank==mpi-1&&electron[i]->x>(Grid_Nx+1)*dx)){
      del_particle_e(electron,i,electron_start);
    }
  }
  
#pragma omp parallel for private(p,j)
  for(i=0;i<THREAD_N;i++){
    electron[i]=electron_start[i];
    while(electron[i]->next_particle!=NULL){
      electron[i]=electron[i]->next_particle;
    }
    
    for(j=0;j<(N*Np)/THREAD_N;j++){
      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	p->x=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Nx+2)*dx);
	p->z=gsl_ran_flat(rnd_p[i],2*dx,3*dx);
	p->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	//p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	p->vy=0.;
	p->vz=V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	p->n=N0e;
	p->flag=1;
	  
	electron[i]->next_particle=p;
	electron[i]->next_particle->prev_particle=electron[i];
	electron[i]=electron[i]->next_particle;
	electron[i]->next_particle=NULL;
      }else{
	printf("Can't allocalte memory\n");
	exit(0);
      }
    }

    for(j=0;j<(N*Np)/THREAD_N;j++){
      if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	p->x=gsl_ran_flat(rnd_p[i],2*dx,(Grid_Nx+2)*dx);
	p->z=gsl_ran_flat(rnd_p[i],(Grid_Nz+1)*dx,(Grid_Nz+2)*dx);
	p->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	//p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	p->vy=0.;
	p->vz=V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	p->n=N0e;
	p->flag=1;
	  
	electron[i]->next_particle=p;
	electron[i]->next_particle->prev_particle=electron[i];
	electron[i]=electron[i]->next_particle;
	electron[i]->next_particle=NULL;
      }else{
	printf("Can't allocalte memory\n");
	exit(0);
      }
    }
  }

  if(myrank==0){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      
      for(j=0;j<((Step_p-2)*Np)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],2*dx,3*dx);
	  p->z=gsl_ran_flat(rnd_p[i],3*dx,(Grid_Nz+1)*dx);
	  p->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	  //p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	  p->vy=0.;
	  p->vz=V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	  p->n=N0e;
	  p->flag=1;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  printf("Can't allocalte memory\n");
	  exit(0);
	}
      }   
    }
  }

  if(myrank==mpi-1){
#pragma omp parallel for private(p,j)
    for(i=0;i<THREAD_N;i++){
      
      for(j=0;j<((Step_p-2)*Np)/THREAD_N;j++){
	if((p=(Particle*)malloc(sizeof(Particle)))!=NULL){
	  p->x=gsl_ran_flat(rnd_p[i],(Grid_Nx+1)*dx,(Grid_Nx+2)*dx);
	  p->z=gsl_ran_flat(rnd_p[i],3*dx,(Grid_Nz+1)*dx);
	  p->vx=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	  //p->vy=gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	  p->vy=0.;
	  p->vz=V+gsl_ran_gaussian(rnd_v[i],sqrt(kb*Te/me));
	  p->n=N0e;
	  p->flag=1;
	  
	  electron[i]->next_particle=p;
	  electron[i]->next_particle->prev_particle=electron[i];
	  electron[i]=electron[i]->next_particle;
	  electron[i]->next_particle=NULL;
	}else{
	  printf("Can't allocalte memory\n");
	  exit(0);
	}
      }   
    }
  }
  return(0);
}

