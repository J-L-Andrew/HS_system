#ifndef _VOXEL_H
#define _VOXEL_H


#include "Superball.h"
#include "algebra.h"

class Voxel {
 public:
  Voxel();
  Voxel(double box, double voxel_l, int nx);
  ~Voxel();

  Vector<3> center;  // center coordinates
  Vector<3> r;       // for indicator function

  double box_l;
  double dl;
  double vol;

  int n;  // number of voxel in each side
  int num_total;
  bool state;  // (1: filled, 0: empty)

 public:
  bool isContain(CSuperball *p);
};

inline int sgn(double x) {
  int t;
  if (x == 0)
    t = 0;
  else if (x > 0)
    t = 1;
  else
    t = -1;
  return t;
}

inline double SuperballFunction(double p, double r[3], Vector<3> X) {
  return exp((double)2.0 * p * log(fabs(X[0]) / r[0])) +
         exp((double)2.0 * p * log(fabs(X[1]) / r[1])) +
         exp((double)2.0 * p * log(fabs(X[2]) / r[2]));
}

// Calculate the Fourier transform of the indicator function for particle j
void FFT_if(CSuperball* p, double resolution, Vector<3> K_t, double* re,
            double* im);

#endif
