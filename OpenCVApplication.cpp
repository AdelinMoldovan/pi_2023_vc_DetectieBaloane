// OpenCVApplication.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include "common.h"
#include "Test.h"
#include <iostream>
#include <fstream>
#include <vector>

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("opened image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat_<Vec3b> src = imread(fname, IMREAD_COLOR);

		int height = src.rows;
		int width = src.cols;

		Mat_<uchar> dst(height, width);

		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("original image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testFindBalloons()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src;
		src = imread(fname, IMREAD_COLOR);
		const auto baloons = getBalloons(fname);
		drawBalloons(src, baloons);
		imshow("opened image", src);
		waitKey();
	}
}
void detectLines() {
	// Încarcă imaginea
	Mat image;
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		image = imread(fname, IMREAD_ANYCOLOR);
		// Aplică filtrul Gaussian
		Mat blurred;
		GaussianBlur(image, blurred, Size(5, 5), 0);

		// Detectează marginile cu algoritmul Canny
		Mat edges;
		Canny(blurred, edges, 50, 150);

		// Afișează imaginea inițială și marginile detectate
		imshow("Imaginea initiala", image);
		imshow("Marginile detectate", edges);
		waitKey(0);
	}
}


int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Basic image opening...\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Color to Gray\n");
		printf(" 4 - TestBalloons\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testColor2Gray();
				break;
			case 4:
				testFindBalloons();
				break;
		}
	}
	while (op!=0);
	return 0;
}
