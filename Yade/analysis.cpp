#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdlib.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

class Point {
 public:
  float x, y;
  Point(float x = 0, float y = 0) : x(x), y(y){};
  float getX() { return x; }
  float getY() { return y; }
};

Point ptemp[10000], p[10000];

int readfile(string dir);
void linearFit(Point points[], int n, double* para);

ofstream kd("fit.txt");

int main() {
  double para[3];
  string infile = "modulus_result.txt";
  int num = readfile(infile);
  linearFit(p, num, &para[0]);

  kd << "k\tb" << endl;
  kd << para[0] << "\t" << para[1] << endl;
  return 0;
}

int readfile(string dir) {
  int num = 0, a;
  string temp;

  ifstream in;
  in.open(dir);

  if (!in.is_open())  // unable to open file
  {
    cout << "Unable to open myfile" << endl;
    exit(1);
  }

  while ((a = in.peek()) != EOF) {
    in >> p[num].x >> p[num].y;
    num += 1;
  }

  return num;
}

void linearFit(Point points[], int n, double* para) {
  float avgX = 0.0, avgY = 0.0;
  float Lxx = 0.0, Lyy = 0.0, Lxy = 0.0;
  float A = 0.0, B = 0.0, C = 0.0, D = 0.0;

  for (int i = 0; i < n; ++i) {
    A += points[i].getX() * points[i].getX();
    B += points[i].getX();
    C += points[i].getX() * points[i].getY();
    D += points[i].getY();
  }

  float b = (C * n - B * D) / (A * n - B * B);
  float a = (A * D - C * B) / (A * n - B * B);

  cout << n << endl;
  cout << A * n - B * B << endl;
  cout << "*--Linear fitting--*" << endl;
  cout << "y=" << b << "x"
       << "+" << a << endl;

  para[0] = b;
  para[1] = a;
}