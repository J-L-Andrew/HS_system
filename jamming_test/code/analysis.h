#include <iostream>
#include <istream>
#include <sstream>
#include <vector>

#include "Cell.h"
#include "Voxel.h"
#include "fstream"
#include "iomanip"
#include "string"

using namespace std;

#define ERR 2.0E-10

class Point {
 public:
  float x, y;
  Point(float x = 0, float y = 0) : x(x), y(y){};
  float getX() { return x; }
  float getY() { return y; }
};

int read_packing(string filename);
int read_moduli(string filename);

void linearFit(Point points[], int n, double *para);

void askforspace();
void releasespace();

void spectral_density();

void getsurpoint(double para[], int num, Vector<3> *SurfPot);
void OutputVoroPoint(int num, int replica);
void VoroVolume(int num, int replica);
void OutputBoundary(string filename, int replica);

void PeriodicCheck(Vector<3> &point);

void POV_superball(string filename);
void XYZ_output(string filename);

double GetCN(int N);
double CalculateQ6();



// ----------------
// Background grid
// ----------------
void PeriodicalCheck(CSuperball *ps);
void BackGroundGrids();
void UpdateGrids();
void AddToBlocks(CSuperball *ps);
void DeleteNode(int blocknumber, int headnumber, CNode *pn);