#include <stdlib.h>
#include <stdio.h>
# include <math.h>
# include <omp.h>
# include <mpi.h>
#include <glut.h>


#define ITERATION 1000
# define TIMESTEP 250
FILE * pFile;
int status = 0; 
int i = 0;
int j = 0;
int n = 0;
int l = 0;
int pos = 0,rankcount,counter,rank,size,offset = 0,divide = 0,npart,extra;
double FX,FY,AX,AY,vx,vy;
int * mass;
double * posx;
double * posy;
double * ux;
double * uy;
double * sx;
double * sy;
double *posx_npart,*posy_npart,*ux_npart,*uy_npart;
int *displacement,*count;
double G =  6.67428E-11;
int iteration = 0;



void Redraw() {
	MPI_Bcast(posx,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(posy,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(ux,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(uy,n,MPI_DOUBLE,0,MPI_COMM_WORLD);

	for( i = displacement[rank]; i < displacement[rank]+npart; i++)
		{
			FX= 0;FY= 0;AX = 0;AY=0;vx = 0;vy =0;
			for (int j = 0; j < n; j++)
			{
				if(i == j)
					continue;
				double deltaX = (posx[j] - posx[i]);
				double deltaY = (posy[j] - posy[i]);
				double dis = sqrt((deltaX*deltaX)+(deltaY*deltaY));
				FX += (G*mass[j]*mass[i]*deltaX)/(dis*dis);//F += G*mn*mi*rni/rni^2
				FY += (G*mass[j]*mass[i]*deltaY)/(dis*dis);
				AX = FX/mass[i];//a = F/m
				AY = FY/mass[i];
			}//end j
			vx = ux[i] + (AX*TIMESTEP);//v = u+at
			vy = uy[i] + (AY*TIMESTEP);
			ux[i] = vx; 
			uy[i] = vy;
			sx[i] = (ux[i]*TIMESTEP) + (0.5*AX*TIMESTEP*TIMESTEP);// s = ut+1/2at^2 
			sy[i] = (uy[i]*TIMESTEP) + (0.5*AY*TIMESTEP*TIMESTEP);

		}//end i
		//CALC new position with the distance moved
		rankcount = 0 ;
		for(pos = displacement[rank]; pos < displacement[rank]+npart; pos++)
		{
			posx[pos] += sx[pos];
			posy[pos] += sy[pos];
			posx_npart[rankcount] = posx[pos];
			posy_npart[rankcount] = posy[pos];
			ux_npart[rankcount] = ux[pos];
			uy_npart[rankcount] = uy[pos];
			rankcount++;
		}
		MPI_Gatherv(posx_npart,npart,MPI_DOUBLE,posx,count,displacement,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Gatherv(posy_npart,npart,MPI_DOUBLE,posy,count,displacement,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Gatherv(ux_npart,npart,MPI_DOUBLE,ux,count,displacement,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Gatherv(uy_npart,npart,MPI_DOUBLE,uy,count,displacement,MPI_DOUBLE,0,MPI_COMM_WORLD);

	}
void display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	for(int i = 0; i < n; i++) 
	 {
		glPointSize(3.0);
        glBegin(GL_POINTS);	
        glColor3f (0.0, 1.0, 0.0);
        glVertex2f(posx[i]*2.5,posy[i]*2.5);//Make the display bigger by *2.5 to the position
        glEnd();
    }

	Redraw();//Update for each iteration LOOP
	glutPostRedisplay();
	iteration++;
	if(iteration == ITERATION)
		exit(0);

	glutSwapBuffers();
	
}
int main(int argc, char **argv)
{

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    glutInit(&argc, argv);
    glutInitDisplayMode((GLUT_DOUBLE | GLUT_RGBA|GLUT_DEPTH ));
    glutInitWindowSize(1920,1080);
	glutInitWindowPosition (0, 0);
    glutCreateWindow("N-Body SIMULATION");
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1000, 1000,0, 0, 500);

	pFile = fopen("inputTest.txt","r");
	if(pFile == NULL)
	{
		printf("Error opening FIle\n");
		exit(0);
	}
	fscanf(pFile,"%d",&n);
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

	mass = (int*) calloc(n ,sizeof(int));
	posx = (double*) calloc(n ,sizeof(double));
	posy = (double*) calloc(n ,sizeof(double));
	ux = (double*) calloc(n ,sizeof(double));
	uy = (double*) calloc(n ,sizeof(double));
	sx = (double*) calloc(n ,sizeof(double));
	sy = (double*) calloc(n ,sizeof(double));

	for(l = 0; l < n; l++)
		fscanf(pFile,"%lf %lf %d",&posx[l],&posy[l],&mass[l]);

	posx_npart = (double*)calloc(npart,sizeof(double));
	posy_npart = (double*)calloc(npart,sizeof(double));
	ux_npart = (double*)calloc(npart,sizeof(double));
	uy_npart = (double*)calloc(npart,sizeof(double));

	glutDisplayFunc(display);
	glutMainLoop();
    return 0;
}


