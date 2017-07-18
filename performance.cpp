#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>


#define TIMESTEP 250
#define ITERATION 1000

int main(int argc, char  **argv)
{	
	FILE * pFile = NULL;
	FILE* pFileout = NULL ;
	double G = 6.67428E-11;
	double * posx,* posy,*posx_npart,*posy_npart,*ux_npart,*uy_npart;
	int * mass;
	double FX,FY,vx,vy,deltaX = 0,deltaY = 0,dis = 0;
	double *sx,*sy,*ux,*uy;
	double StartTime = 0,EndTime = 0,StartTime1 = 0;
	int *displacement,*count;
	int i = 0,j,k,l = 0,pos = 0,rankcount,n,counter,rank,size,offset = 0,divide = 0,npart,extra;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	omp_set_num_threads(8);
	


	pFile = fopen("inputTest.txt","r");
	if (pFile == NULL)
	{
		printf("Error on file, exiting\n");
		exit(0);
	}
	fscanf(pFile,"%d",&n);


	//Allocate for the main datatypes
	posx = (double*) calloc(n ,sizeof(double));
	posy = (double*) calloc(n ,sizeof(double));
	mass = (int*) calloc(n ,sizeof(int));
	ux = (double*) calloc(n ,sizeof(double));
	uy = (double*) calloc(n ,sizeof(double));
	sx = (double*) calloc(n ,sizeof(double));
	sy = (double*) calloc(n ,sizeof(double));

	//Divide n number of tasks equally among processors, if there are extra then add each extra task to each processor respectively
	divide = n/size;
	extra =n%size;
	npart = (rank < extra) ? divide+1:divide;
	displacement = (int*) calloc(size,sizeof(int));
	count = (int*) calloc(size,sizeof(int));
	for( i=0;i<size;i++)
		{	 
		counter = ((i<extra)?divide+1:divide); 
		displacement[i]=offset;
		count[i]=counter;
		offset=offset+counter;			
		}


	for (i = 0; i < n; i++)
		fscanf(pFile,"%lf %lf %d",&posx[i],&posy[i],&mass[i]);		

	pFileout = fopen("output.txt","w");

	// Allocate memory for each processors divided calculation
	posx_npart = (double*)calloc(npart,sizeof(double));
	posy_npart = (double*)calloc(npart,sizeof(double));
	ux_npart = (double*)calloc(npart,sizeof(double));
	uy_npart = (double*)calloc(npart,sizeof(double));  

	#pragma omp parallel for default (shared) private(i,j,k) shedule(static) reduction(+: FX,FY)
	for (k = 0; k < ITERATION; k++)
	{
	StartTime = MPI_Wtime();
		// Update the new values for each processor after every gather for each iteration.
		MPI_Bcast(posx,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(posy,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(ux,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(uy,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
		//Calculate each calculation for each processor with new displacement for each processor.
		for ( i = displacement[rank]; i < displacement[rank]+npart; i++)
		{	
			FX = 0;FY = 0;vx = 0;vy = 0;
			for ( j = 0; j < n; j++)
			{
				
				if (i == j) /*Avoid find force from same particle*/
					continue;
				deltaX = posx[j] - posx[i];
				deltaY = posy[j] - posy[i];
				dis = sqrt((deltaX*deltaX) + (deltaY*deltaY));
				if(dis == 0)
					continue;
				FX += (G*mass[j]*mass[i]*deltaX)/(dis*dis);//F += G*mn*mi*rni/rni^2
				FY += (G*mass[j]*mass[i]*deltaY)/(dis*dis);
			}// End j
			vx = ux[i] + (FX/mass[i])*TIMESTEP;//v = u+at
			vy = uy[i] + (FY/mass[i])*TIMESTEP;
			ux[i] = vx;
			uy[i] = vy;
			sx[i] = ux[i]*TIMESTEP + 0.5*(FX/mass[i]) * TIMESTEP * TIMESTEP;// s = ut+1/2at^2
			sy[i] = uy[i]*TIMESTEP + 0.5*(FY/mass[i]) * TIMESTEP * TIMESTEP;
	
		} // End i

		//CALC new position with the distance moved for each processor
		rankcount = 0 ;//start the saving from 0 for each processor
		for (pos = displacement[rank]; pos < displacement[rank]+npart; pos++)
		{
			posx[pos] += sx[pos]; posy[pos] += sy[pos];
			ux_npart[rankcount] = ux[pos]; uy_npart[rankcount] = uy[pos];
			posx_npart[rankcount] = posx[pos]; posy_npart[rankcount] = posy[pos];
			rankcount++;
		}
		// Processor 0 gathers each divided calculated part from each processor
		MPI_Gatherv(posx_npart,npart,MPI_DOUBLE,posx,count,displacement,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Gatherv(posy_npart,npart,MPI_DOUBLE,posy,count,displacement,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Gatherv(ux_npart,npart,MPI_DOUBLE,ux,count,displacement,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Gatherv(uy_npart,npart,MPI_DOUBLE,uy,count,displacement,MPI_DOUBLE,0,MPI_COMM_WORLD);
		EndTime = MPI_Wtime();
		StartTime1 = (EndTime - StartTime)+StartTime1;
		if(rank ==0)
		{
			fprintf(pFileout,"New Iteration\n");
			for(i = 0;i<n;i++)
				fprintf(pFileout,"%.2lf %.2lf %d \n",posx[i],posy[i],mass[i]);
		}
		
		
	}// End k

	
	if (rank == 0)
	{
		printf("%d\n",n);
		for(i = 0;i<n;i++)
			printf("%.2lf %.2lf %d \n",posx[i],posy[i],mass[i]);
		printf("TIME %.2lf", StartTime1);
	}

	MPI_Finalize();
}