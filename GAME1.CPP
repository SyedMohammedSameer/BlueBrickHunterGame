#include<stdio.h>
#include<conio.h>
#include<graphics.h>
#include<dos.h>
#include<stdlib.h>
#define MOVE_LEFT 75
#define MOVE_RIGHT 77
#define ESC 27
char ch;
void print_line();
void print_line2();
void messagebox();
void winner();
void main()
{
	int gd=DETECT,gm,i,j,IMG1,IMG2,IMG3,x,y,win=0;
	void *BUFFER1,*BUFFER2,*BUFFER3;
	int x_axis=325,y_axis=325,ch,x_direction=-1,y_direction=-1,speed=15;
	int X_POSITION=0,Y_POSITION=0;
	registerfarbgidriver(EGAVGA_driver_far);
	registerfarbgifont(sansserif_font_far);
	registerfarbgifont(small_font_far);
	registerfarbgifont(gothic_font_far);
	registerfarbgifont(triplex_font_far);

	initgraph(&gd,&gm,"C:\\TURBOC3\\BGI");
	cleardevice();
	setbkcolor(2);

	rectangle(0,0,60,22);
	setfillstyle(1,9);
	floodfill(2,2,15);
	IMG1=imagesize(0,0,60,22);
	BUFFER1=malloc(IMG1);
	getimage(0,0,60,22,BUFFER1);
	putimage(0,0,BUFFER1,XOR_PUT);
	for(j=0;j<300;j+=30)
	for(i=8;i<600;i+=70)
	{
	putimage(i,30+j,BUFFER1,XOR_PUT);
	}

	setcolor(6);
	rectangle(80,445,180,460);
	setfillstyle(1,15);
	floodfill(82,447,6);
	IMG2=imagesize(80,445,180,460);
	BUFFER2=malloc(IMG2);
	getimage(80,445,180,460,BUFFER2);

	setcolor(WHITE);
	setfillstyle(1,12);
	circle(350,350,10);
	floodfill(350,350,WHITE);
	print_line();
	print_line2();

	IMG3=imagesize(340,340,360,360);
	BUFFER3=malloc(IMG3);
	getimage(340,340,360,360,BUFFER3);
	putimage(340,340,BUFFER3,XOR_PUT);

	for(;;)
	{
	putimage(x_axis,y_axis,BUFFER3,XOR_PUT);
	delay(speed);
	putimage(x_axis,y_axis,BUFFER3,XOR_PUT);

	x_axis=x_axis+(x_direction*2);
	y_axis=y_axis+(y_direction*3);
	if(x_axis+10-1>640)
	{
	x_direction*=-1;
	x_axis=640-10+1;
	}
	if(x_axis<0)
	{
	x_direction*=-1;
	x_axis=0;
	}
	if(y_axis+10-1>470)
	{
	cleardevice();
	initgraph(&gd,&gm,"C:\\TURBOC3\\BGI");
	outtextxy(200,200,"HEY YOU LOSE THE GAME");

	free(BUFFER1);
	free(BUFFER2);
	free(BUFFER3);
	delay(2000);
	messagebox();
	}

	if(getpixel(x_axis,y_axis)==6)
	{
	sound(900);
	delay(15);
	nosound();
	y_direction*=-1;
	}
	if(getpixel(x_axis,y_axis)==9)
	{
	sound(300);
	delay(50);
	nosound();
	y_direction*=-1;
	x=x_axis;
	y=y_axis;
	x=(x/70)*70+8;
	y=(y/30)*30;
	putimage(x,y,BUFFER1,XOR_PUT);
	win++;
	if(win==40)
	{
	winner();
	}
	}
	if(y_axis<0)
	{
	y_direction*=-1;
	y_axis=0;
	}

	if(kbhit())
	{
	ch=getch();
	if(ch!=ESC)
	{
	putimage(80+X_POSITION,445+Y_POSITION,BUFFER2,XOR_PUT);
	}
	if(ch==MOVE_LEFT)
	X_POSITION+=-40;
	if(ch==MOVE_RIGHT)
	X_POSITION+=40;
	putimage(80+X_POSITION,445+Y_POSITION,BUFFER2,XOR_PUT);

	}
	if(ch==ESC)
	{
	free(BUFFER1);
	free(BUFFER2);
	free(BUFFER3);
	closegraph();
	exit(0);
	}
	}
	}



	void print_line()
	{
	setcolor(WHITE);
	line(0,18,640,18);
	line(0,19,640,19);
	line(0,20,640,20);
	line(0,21,640,21);
	line(0,22,640,22);
	}
	void print_line2()
	{
	line(0,465,640,465);
	line(0,466,640,466);
	line(0,467,640,467);
	line(0,468,640,468);
	}
	void messagebox()
	{
	char ch;
	cleardevice();
	setcolor(WHITE);
	rectangle(178,108,458,208);
	rectangle(177,107,459,209);
	rectangle(176,106,460,210);
	rectangle(175,105,461,211);
	settextstyle(11,0,1);
	nosound();
	setcolor(12);

	outtextxy(222,140,"DO YOU WANT TO PLAY AGAIN ?");
	setcolor(11);
	outtextxy(222,160,"PRESS [Y] OR [N]!");
	setcolor(15);
	outtextxy(310,190,"CODE BY SAMEER");

	ch=getch();
	if(ch=='Y')
	{
	main();
	}
	else
	{
	nosound();
	exit(0);
	}
	}
	void winner()
	{
	cleardevice();
	setbkcolor(0);
	setcolor(WHITE);
	rectangle(179,109,457,207);
	rectangle(178,108,458,208);
	rectangle(177,107,459,209);
	rectangle(176,106,460,210);
	rectangle(175,105,461,211);
	setcolor(14);
	outtextxy(222,140,"HEY YOU WON THE GAME!");
	sound(3333);
	delay(999);
	nosound();
	delay(200);
	exit(0);
	}