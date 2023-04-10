#include "Superball.h"

#include <stdlib.h>

CSuperball::CSuperball(void) { bl = NULL, hl = NULL, nl = NULL; }
CSuperball::CSuperball(double x) {
  bl = NULL, hl = NULL, nl = NULL;
  if (!neigh_list.empty()) neigh_list.clear();

  p = x;

  r_scale[0] = 1.0;
  r_scale[1] = 1.0;
  r_scale[2] = 1.0;

  e[0] = {1, 0, 0};
  e[1] = {0, 1, 0};
  e[2] = {0, 0, 1};

  double a = tgamma(0.5 / p);
  double b = tgamma(1.5 / p);
  vol = 2.0 * a * a * a / p / p / b / 3.0;

  double r = sqrt(3.0) * pow(1.0 / 3.0, 0.5 / p);
  if (r > 1.0)
    bound_D = 2.0 * r;
  else
    bound_D = 2.0;

  d_v = pow(3.0 * vol / 4.0 / PI, 1.0 / 3.0);
}

CSuperball::CSuperball(double x, double sx, double sy, double sz) {
  bl = NULL, hl = NULL, nl = NULL;
  if (!neigh_list.empty()) neigh_list.clear();

  p = x;
  r_scale[0] = sx;
  r_scale[1] = sy;
  r_scale[2] = sz;

  e[0] = {1, 0, 0};
  e[1] = {0, 1, 0};
  e[2] = {0, 0, 1};

  double a = tgamma(0.5 / p);
  double b = tgamma(1.5 / p);
  vol = 2.0 * a * a * a / p / p / b / 3.0;
  vol *= (sx * sy * sz);

  double r;
  double b_x, b_y, b_z;
  double temp;

  if (p > 0.99999 && p < 1.00001)  // ellipsoid
  {
    temp = sx;
    if (sy > temp) temp = sy;
    if (sz > temp) temp = sz;
    bound_D = 2.0 * temp;
  } else if (p >= 1.00001) {
    temp = pow(0.5, 0.5 / p);
    b_x = sx * temp;
    b_y = sy * temp;
    b_z = sz;
    r = b_x * b_x + b_y * b_y;

    bound_D = 2.0 * sqrt(r + b_z * b_z);
    // for only superball  sx=sy=sz
    // temp = sx * pow(1.0 / 3.0, 0.5 / p);
    // bound_D = 2.0 * sqrt(3.0) * temp;
  } else {
    b_x = sx;
    b_y = sy;
    b_z = sz;
    temp = b_x;
    if (b_y > temp) temp = b_y;

    if (temp > b_z)
      bound_D = 2.0 * temp;
    else
      bound_D = 2.0 * b_z;
  }

  d_v = pow(3.0 * vol / 4.0 / PI, 1.0 / 3.0);
}

CSuperball::~CSuperball() {
  if (bl != NULL) delete[] bl;
  if (hl != NULL) delete[] hl;
  if (nl != NULL) delete[] nl;
}

void CSuperball::Destructor(void) {}

void CSuperball::Copyfrom(
    CSuperball* psp)  // only copy informations used in distance function
{
  int i;
  p = psp->p;
  for (i = 0; i < 3; i++) r_scale[i] = psp->r_scale[i];
  center = psp->center;
  for (i = 0; i < 3; i++) e[i] = psp->e[i];
  vol = psp->vol;
  bound_D = psp->bound_D;
}
void CSuperball::Update() {}

void CSuperball::Boundary(double X[2], double Y[2], double Z[2]) {
  X[0] = X[1] = center[0];
  Y[0] = Y[1] = center[1];
  Z[0] = Z[1] = center[2];

  X[0] -= 0.5 * bound_D, X[1] += 0.5 * bound_D;
  Y[0] -= 0.5 * bound_D, Y[1] += 0.5 * bound_D;
  Z[0] -= 0.5 * bound_D, Z[1] += 0.5 * bound_D;
}
void CSuperball::Jump(double PBC[3]) {
  for (int i = 0; i < 3; i++) {
    center[i] += PBC[i];
  }
}
void CSuperball::Translate() {
  center[0] += TV[0];
  center[1] += TV[1];
  center[2] += TV[2];
}
void CSuperball::Translate(double a) {
  center[0] += TV[0] * a;
  center[1] += TV[1] * a;
  center[2] += TV[2] * a;
}
void CSuperball::Translate(Vector<3> V) { center += V; }
void CSuperball::Rotate(double a) {
  Vector<3> u;
  double b, cosb, sinb;

  b = RV[0] * RV[0] + RV[1] * RV[1] + RV[2] * RV[2];
  if (b < 1.0E-15) return;
  b = sqrt(b);
  u = RV / b;
  b *= a;

  cosb = cos(b);
  sinb = sin(b);
  e[0] = e[0] * cosb + u * (e[0].Dot(u)) * (1.0 - cosb) + u.Cross(e[0]) * sinb;
  e[1] = e[1] * cosb + u * (e[1].Dot(u)) * (1.0 - cosb) + u.Cross(e[1]) * sinb;
  e[2] = e[0].Cross(e[1]);
}
void CSuperball::Rotate(Vector<3> V, double a) {
  if (a < 1.0E-30) return;
  Vector<3> u;
  double sina, cosa;
  u = V;
  u.Normalize();
  sina = sin(a);
  cosa = cos(a);
  e[0] = e[0] * cosa + u * (e[0].Dot(u)) * (1.0 - cosa) + u.Cross(e[0]) * sina;
  e[1] = e[1] * cosa + u * (e[1].Dot(u)) * (1.0 - cosa) + u.Cross(e[1]) * sina;
  e[2] = e[0].Cross(e[1]);
}
void CSuperball::Randomize(double a) {
  double b;
  Vector<3> X, V;

  center = {a * (rand() + 1.0) / (RAND_MAX + 2.0),
            a * (rand() + 1.0) / (RAND_MAX + 2.0),
            a * (rand() + 1.0) / (RAND_MAX + 2.0)};
  do {
    X = {2.0 * rand() / (RAND_MAX + 1.0) - 1.0,
         2.0 * rand() / (RAND_MAX + 1.0) - 1.0,
         2.0 * rand() / (RAND_MAX + 1.0) - 1.0};
    // X.Set(0.0,0.0,0.5);
    b = X.Norm();
  } while (b < 0.000001 || b > 1.0);
  X.Normalize();
  V = e[0].Cross(X);
  b = e[0].Dot(X);
  if (b > 1.0)
    b = 1.0;
  else if (b < -1.0)
    b = -1.0;
  Rotate(V, acos(b));
  Rotate(X, 2.0 * PI * rand() / (RAND_MAX + 1.0));
}
void CSuperball::Randomize2(double a, Vector<3> X, double betra) {
  double b;
  Vector<3> V;

  center = {a * (rand() + 1.0) / (RAND_MAX + 2.0),
            a * (rand() + 1.0) / (RAND_MAX + 2.0),
            a * (rand() + 1.0) / (RAND_MAX + 2.0)};
  b = X.Norm();
  X.Normalize();
  V = e[0].Cross(X);
  b = e[0].Dot(X);
  if (b > 1.0)
    b = 1.0;
  else if (b < -1.0)
    b = -1.0;
  Rotate(V, acos(b));
  Rotate(X, betra * PI / 180.0);
}
void CSuperball::Set(double a, int i) {
  double b;
  Vector<3> X, V;

  center = {0.5 * a * (double(i) + 0.1), 0.5 * a * (double(i) + 0.1),
            0.5 * a * (double(i) + 0.1)};
  do {
    X = {0.5 * double(i), 0.5, 0.5 * double(i)};
    b = X.Norm();
  } while (b < 0.000001 || b > 1.0);
  X.Normalize();
  V = e[0].Cross(X);
  b = e[0].Dot(X);
  if (b > 1.0)
    b = 1.0;
  else if (b < -1.0)
    b = -1.0;
  Rotate(V, acos(b));
  Rotate(X, 0.3 * PI);
}

void CSuperball::RandomizeOrientation() {
  double b;
  Vector<3> X, V;
  do {
    X = {2.0 * rand() / (RAND_MAX + 1.0) - 1.0,
         2.0 * rand() / (RAND_MAX + 1.0) - 1.0,
         2.0 * rand() / (RAND_MAX + 1.0) - 1.0};
    // X.Set(0.0,0.0,0.5);
    b = X.Norm();
  } while (b < 0.000001 || b > 1.0);
  X.Normalize();
  V = e[0].Cross(X);
  b = e[0].Dot(X);
  if (b > 1.0)
    b = 1.0;
  else if (b < -1.0)
    b = -1.0;
  Rotate(V, acos(b));
  Rotate(X, 2.0 * PI * rand() / (RAND_MAX + 1.0));
}