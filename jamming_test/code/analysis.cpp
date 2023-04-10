#include "analysis.h"
#define Maxlocal 12

int DIM;
int NUM_PARTICLE;
Matrix<3, 3> Lambda;  // Lattice vectors for unit cell

double sumvol;
double PD;

int Nkind;
int *NpEach;
double *Diameter;

int MASK3D[14][2] = {0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6,
                     0, 7, 1, 2, 1, 4, 1, 6, 2, 4, 2, 5, 3, 4};

double L;         // Box length
double l = 2.35;  // length of (Background) grid cell
int nb;           // Number of grid in each direction
int NNN;          // = nb*nb*nb;

CSuperball *psp[2];
double **pair_dis, **neigh_dmin;
int **neigh_id;
Vector<3> **pair_dv;

vector<CCell *> blocks;
vector<CSuperball *> particles;

Point p[10000];

// ofstream kd("fit.txt");

int main() {
  string head = "/mnt/Edisk/andrew/dataset/LS/new/sample";
  string file;

  // ofstream sum("./sum.txt");
  // sum << "PD\tG\tR^2" << endl;

  for (int i = 1320; i <= 1320; ++i) {
    double para[3];
    string index;
    stringstream trans;
    int count = i;

    trans << count;
    trans >> index;
    string path = head + index;

    int flag = read_packing(path + "/final.dat");
    // int num = read_moduli(path + "/modulus_result.txt");
    // linearFit(p, num, &para[0]);

    // kd << "k\tb" << endl;
    // kd << para[0] << "\t" << para[1] << endl;

    if (flag) {
      // sum << PD << "\t" << para[0] << "\t" << para[1] << endl;
      BackGroundGrids();

      double Q6 = CalculateQ6_local();
      cout << Q6 << endl;
      POV_superball(path + "/config.pov");
      XYZ_output(path + "/config.xyz");
      releasespace();
    }
  }

  // // OutputBoundary("../vorofile/boundary.txt", 1);
  // // OutputVoroPoint(20, 1);
  // VoroVolume(20, 1);
  // spectral_density();
  return 0;
}

// ----------------------------------------------------------------
// Ask and release space
// ----------------------------------------------------------------
void askforspace() {
  NpEach = new int[Nkind];
  Diameter = new double[Nkind];
  pair_dis = new double *[NUM_PARTICLE];

  psp[0] = new CSuperball();
  psp[1] = new CSuperball();

  pair_dis = new double *[NUM_PARTICLE];
  for (int i = 0; i < NUM_PARTICLE; i++) pair_dis[i] = new double[NUM_PARTICLE];
  pair_dv = new Vector<3> *[NUM_PARTICLE];
  for (int i = 0; i < NUM_PARTICLE; i++)
    pair_dv[i] = new Vector<3>[NUM_PARTICLE];
  neigh_id = new int *[NUM_PARTICLE];
  for (int i = 0; i < NUM_PARTICLE; i++) neigh_id[i] = new int[NUM_PARTICLE];
  neigh_dmin = new double *[NUM_PARTICLE];
  for (int i = 0; i < NUM_PARTICLE; i++) neigh_dmin[i] = new double[Maxlocal];
}
void releasespace() {
  delete[] NpEach;
  delete[] Diameter;

  delete psp[0];
  delete psp[1];

  for (int i = 0; i < NUM_PARTICLE; i++) delete[] pair_dis[i];
  delete[] pair_dis;
  for (int i = 0; i < NUM_PARTICLE; i++) delete[] pair_dv[i];
  delete[] pair_dv;
  for (int i = 0; i < NUM_PARTICLE; i++) delete[] neigh_id[i];
  delete[] neigh_id;
  for (int i = 0; i < NUM_PARTICLE; i++) delete[] neigh_dmin[i];
  delete[] neigh_dmin;

  blocks.clear();
  particles.clear();
}

// ----------------------------------------------------------------
// Read packing
// ----------------------------------------------------------------
int read_packing(string filename) {
  if (!particles.empty()) {
    for (int i = 0; i < particles.size(); i++) delete particles[i];
    particles.clear();
  }

  ifstream myfile(filename);

  if (!myfile.is_open()) {
    cout << "Unable to open myfile" << endl;
    // exit(1);
    cout << filename << endl;
    return 0;
  }

  string temp;

  myfile >> DIM;
  myfile >> NUM_PARTICLE >> Nkind;

  askforspace();
  for (int i = 0; i < Nkind; ++i) {
    myfile >> NpEach[i];
  }
  for (int i = 0; i < Nkind; ++i) {
    myfile >> Diameter[i];
  }

  myfile >> Lambda[0][0] >> Lambda[0][1] >> Lambda[0][2] >> Lambda[1][0] >>
      Lambda[1][1] >> Lambda[1][2] >> Lambda[2][0] >> Lambda[2][1] >>
      Lambda[2][2];

  L = Lambda[0][0];

  getline(myfile, temp);
  getline(myfile, temp);

  double radius = Diameter[0] / 2.0;
  sumvol = 0.0;
  for (int i = 0; i < NUM_PARTICLE; ++i) {
    particles.push_back(new CSuperball(1.0, radius, radius, radius));
    particles[i]->ID = i;
    sumvol += particles[i]->vol;

    myfile >> particles[i]->center[0] >> particles[i]->center[1] >>
        particles[i]->center[2];
  }

  PD = sumvol / Lambda.Determinant();
  return 1;
}

// ----------------------------------------------------------------
// Yade module (readfile and fitting)
// ----------------------------------------------------------------
int read_moduli(string dir) {
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
void linearFit(Point points[], int n, double *para) {
  float avgX = 0.0, avgY = 0.0;
  float Lxx = 0.0, Lyy = 0.0, Lxy = 0.0;
  float A = 0.0, B = 0.0, C = 0.0, D = 0.0, E = 0.0;

  for (int i = 0; i < n; ++i) {
    A += points[i].getX() * points[i].getX();
    B += points[i].getX();
    C += points[i].getX() * points[i].getY();
    D += points[i].getY();
    E += points[i].getY() * points[i].getY();
  }

  float b = (C * n - B * D) / (A * n - B * B);
  float a = (A * D - C * B) / (A * n - B * B);

  // cout << "*--Linear fitting--*" << endl;
  // cout << "y=" << b << "x"
  //      << "+" << a << endl;

  float S_tot = E - D * D / n;
  float S_reg = D * D / n - 2.0 * D / n * (b * B + n * a) + b * b * A +
                2.0 * b * a * B + n * a * a;
  float S_res =
      b * b * A + 2.0 * b * a * B - 2.0 * b * C - 2.0 * a * D + n * a * a + E;
  float R_square = 1.0 - S_res / S_tot;

  para[0] = b;
  para[1] = R_square;
}

// ----------------------------------------------------------------
// Structure property analysis
// ----------------------------------------------------------------
void sort_dis(int id, int index) {
  double minvalue = INFINITY;
  int markn = -1;
  for (int i = 0; i < NUM_PARTICLE; i++) {
    bool flag = 0;
    for (int j = 0; j < index; j++) {
      if (i == neigh_id[id][j]) {
        flag = 1;
        break;
      }
    }

    if (flag) continue;
    if (pair_dis[id][i] < minvalue) {
      minvalue = pair_dis[id][i];
      markn = i;
    }
  }
  neigh_dmin[id][index] = minvalue;
  neigh_id[id][index] = markn;
}

void Nearest_neighbor() {
  for (int i = 0; i < NUM_PARTICLE; i++) {
    pair_dis[i][i] = INFINITY;
    for (int j = i + 1; j < NUM_PARTICLE; j++) {
      Vector<3> vtemp = particles[j]->center - particles[i]->center;
      double dmin = INFINITY;
      Vector<3> vmin;
      for (int ii = -1; ii < 2; ii++)
        for (int jj = -1; jj < 2; jj++)
          for (int kk = -1; kk < 2; kk++) {
            Vector<3> PBC = {double(ii), double(jj), double(kk)};
            PBC = Lambda * PBC;
            double d = (vtemp + PBC).Norm();
            if (d < dmin) {
              dmin = d;
              vmin = vtemp + PBC;
            }
          }
      pair_dis[i][j] = pair_dis[j][i] = dmin;
      pair_dv[i][j] = vmin;
      pair_dv[j][i] = -vmin;
    }
  }

  for (int m = 0; m < NUM_PARTICLE; m++) {
    for (int n = 0; n < Maxlocal; n++) sort_dis(m, n);
  }
}

// void Nearest_neighbor() {
//   int i, j, *ncn;
//   CNode *pn0, *pn1;
//   CSuperball *ps0, *ps1;

//   for (int m = 0; m < NUM_PARTICLE; m++) {
//     for (int n = m; n < NUM_PARTICLE; n++) {
//       pair_dis[m][n] = pair_dis[n][m] = INFINITY;
//     }
//   }

//   for (int t = 0; t < NNN; t++) {
//     for (int m = 0; m < 14; m++) {
//       i = MASK3D[m][0], j = MASK3D[m][1];
//       for (pn0 = blocks[t]->head[i]; pn0 != NULL; pn0 = pn0->next) {
//         ps0 = pn0->ps;
//         if (pn0->PBC != NULL) {
//           psp[0]->Copyfrom(ps0);
//           psp[0]->Jump(pn0->PBC);
//           ps0 = psp[0];
//         }

//         for ((j == 0) ? (pn1 = pn0->next) : (pn1 = blocks[t]->head[j]);
//              pn1 != NULL; pn1 = pn1->next) {
//           ps1 = pn1->ps;
//           if (pn1->PBC != NULL) {
//             psp[1]->Copyfrom(ps1);
//             psp[1]->Jump(pn1->PBC);
//             ps1 = psp[1];
//           }
//           Vector<3> vtemp = ps1->center - ps0->center;

//           pair_dis[pn0->ps->ID][pn1->ps->ID] =
//               pair_dis[pn1->ps->ID][pn0->ps->ID] = vtemp.Norm();
//           pair_dv[pn0->ps->ID][pn1->ps->ID] = vtemp;
//           pair_dv[pn1->ps->ID][pn0->ps->ID] = -vtemp;
//         }
//       }
//     }
//   }

//   for (int m = 0; m < NUM_PARTICLE; m++) {
//     for (int n = 0; n < Maxlocal; n++) sort_dis(m, n);
//   }
// }

double GetCN(int N) {
  // Here for hard particle
  int i, j, *ncn;
  CNode *pn0, *pn1;
  CSuperball *ps0, *ps1;

  double rcut = 2.008 * particles[0]->r_scale[0];

  for (i = 0; i < NUM_PARTICLE; i++) particles[i]->nol = 0;
  for (int t = 0; t < NNN; t++) {
    for (int m = 0; m < 14; m++) {
      i = MASK3D[m][0], j = MASK3D[m][1];
      for (pn0 = blocks[t]->head[i]; pn0 != NULL; pn0 = pn0->next) {
        ps0 = pn0->ps;
        if (pn0->PBC != NULL) {
          psp[0]->Copyfrom(ps0);
          psp[0]->Jump(pn0->PBC);
          ps0 = psp[0];
        }
        for ((j == 0) ? (pn1 = pn0->next) : (pn1 = blocks[t]->head[j]);
             pn1 != NULL; pn1 = pn1->next) {
          ps1 = pn1->ps;
          if (pn1->PBC != NULL) {
            psp[1]->Copyfrom(ps1);
            psp[1]->Jump(pn1->PBC);
            ps1 = psp[1];
          }

          Vector<3> vtemp = ps0->center - ps1->center;
          if (vtemp.Norm() < rcut) {
            pn0->ps->nol++;
            pn1->ps->nol++;
          }
        }
      }
    }
  }
  ncn = new int[N];
  for (i = 0; i < N; i++) ncn[i] = 0;
  double meancn = 0.0;
  for (i = 0; i < NUM_PARTICLE; i++) {
    if (particles[i]->nol < N) ncn[particles[i]->nol]++;
  }
  cout << "Contact Number Distribution:" << endl;
  for (i = 0; i < N; i++) {
    cout << i << "\t" << ncn[i] << endl;
    meancn += i * ncn[i];
  }
  meancn /= double(NUM_PARTICLE);
  delete[] ncn;
  return meancn;
}
double CalculateQ6() {
  Nearest_neighbor();

  int Nb = Maxlocal;
  double cita, fai, Y6 = 0.0;
  double YA6[13], YB6[13], YAT6[13], YBT6[13],
      YT6[13];  // Y6[-6]到Y6[6]变为Y6[0]到Y6[12], Y6=YA6+YB6*i ,即y=a+bi;

  for (int q = 0; q < 13; q++) {
    YA6[q] = YB6[q] = YAT6[q] = YBT6[q] = YT6[q] = 0.0;
  }

  for (int i = 0; i < NUM_PARTICLE; i++) {
    for (int j = 0; j < Maxlocal; j++) {
      int m = neigh_id[i][j];
      Vector<3> vtemp = pair_dv[i][m];

      cita = acos(vtemp[2] / vtemp.Norm());
      double tmp1 = cos(cita);
      double tmp2 = sin(cita);
      fai = atan2(vtemp[1], vtemp[0]);

      YA6[0] = cos(6 * fai) * pow(tmp2, 6) * sqrt(3003 / PI) / 64;
      YB6[0] = -sin(6 * fai) * pow(tmp2, 6) * sqrt(3003 / PI) / 64;
      YA6[1] = 3 * cos(5 * fai) * pow(tmp2, 5) * tmp1 * sqrt(1001 / PI) / 32;
      YB6[1] = -3 * sin(5 * fai) * pow(tmp2, 5) * tmp1 * sqrt(1001 / PI) / 32;
      YA6[2] = 3 * cos(4 * fai) * pow(tmp2, 4) * (11 * pow(tmp1, 2) - 1) *
               sqrt(91 / PI / 2) / 32;
      YB6[2] = -3 * sin(4 * fai) * pow(tmp2, 4) * (11 * pow(tmp1, 2) - 1) *
               sqrt(91 / PI / 2) / 32;
      YA6[3] = cos(3 * fai) * pow(tmp2, 3) * (11 * pow(tmp1, 3) - 3 * tmp1) *
               sqrt(1365 / PI) / 32;
      YB6[3] = -sin(3 * fai) * pow(tmp2, 3) * (11 * pow(tmp1, 3) - 3 * tmp1) *
               sqrt(1365 / PI) / 32;
      YA6[4] = cos(2 * fai) * pow(tmp2, 2) *
               (33 * pow(tmp1, 4) - 18 * pow(tmp1, 2) + 1) * sqrt(1365 / PI) /
               64;
      YB6[4] = -sin(2 * fai) * pow(tmp2, 2) *
               (33 * pow(tmp1, 4) - 18 * pow(tmp1, 2) + 1) * sqrt(1365 / PI) /
               64;
      YA6[5] = cos(fai) * tmp2 *
               (33 * pow(tmp1, 5) - 30 * pow(tmp1, 3) + 5 * tmp1) *
               sqrt(273 / PI / 2) / 16;
      YB6[5] = -sin(fai) * tmp2 *
               (33 * pow(tmp1, 5) - 30 * pow(tmp1, 3) + 5 * tmp1) *
               sqrt(273 / PI / 2) / 16;
      YA6[6] =
          (231 * pow(tmp1, 6) - 315 * pow(tmp1, 4) + 105 * pow(tmp1, 2) - 5) *
          sqrt(13 / PI) / 32;
      YB6[6] = 0;
      YA6[7] = -cos(fai) * tmp2 *
               (33 * pow(tmp1, 5) - 30 * pow(tmp1, 3) + 5 * tmp1) *
               sqrt(273 / 2 / PI) / 16;
      YB6[7] = -sin(fai) * tmp2 *
               (33 * pow(tmp1, 5) - 30 * pow(tmp1, 3) + 5 * tmp1) *
               sqrt(273 / 2 / PI) / 16;
      YA6[8] = cos(2 * fai) * pow(tmp2, 2) *
               (33 * pow(tmp1, 4) - 18 * pow(tmp1, 2) + 1) * sqrt(1365 / PI) /
               64;
      YB6[8] = sin(2 * fai) * pow(tmp2, 2) *
               (33 * pow(tmp1, 4) - 18 * pow(tmp1, 2) + 1) * sqrt(1365 / PI) /
               64;
      YA6[9] = -cos(3 * fai) * pow(tmp2, 3) * (11 * pow(tmp1, 3) - 3 * tmp1) *
               sqrt(1365 / PI) / 32;
      YB6[9] = -sin(3 * fai) * pow(tmp2, 3) * (11 * pow(tmp1, 3) - 3 * tmp1) *
               sqrt(1365 / PI) / 32;
      YA6[10] = 3 * cos(4 * fai) * pow(tmp2, 4) * (11 * pow(tmp1, 2) - 1) *
                sqrt(91 / PI / 2) / 32;
      YB6[10] = 3 * sin(4 * fai) * pow(tmp2, 4) * (11 * pow(tmp1, 2) - 1) *
                sqrt(91 / PI / 2) / 32;
      YA6[11] = -3 * cos(5 * fai) * pow(tmp2, 5) * tmp1 * sqrt(1001 / PI) / 32;
      YB6[11] = -3 * sin(5 * fai) * pow(tmp2, 5) * tmp1 * sqrt(1001 / PI) / 32;
      YA6[12] = cos(6 * fai) * pow(tmp2, 6) * sqrt(3003 / PI) / 64;
      YB6[12] = sin(6 * fai) * pow(tmp2, 6) * sqrt(3003 / PI) / 64;

      for (int t = 0; t < 13; t++) {
        YAT6[t] += YA6[t];
        YBT6[t] += YB6[t];
      }
    }
  }

  for (int s = 0; s < 13; s++) {
    YAT6[s] /= double(Nb * NUM_PARTICLE);
    YBT6[s] /= double(Nb * NUM_PARTICLE);
  }

  for (int u = 0; u < 13; u++) {
    YT6[u] = YAT6[u] * YAT6[u] + YBT6[u] * YBT6[u];
    Y6 += YT6[u];
  }

  return sqrt(4.0 * PI * Y6 / 13.0);
}

double CalculateQ6_local() {
  Nearest_neighbor();

  int Nb = Maxlocal;
  double cita, fai, Y6 = 0.0;
  double YA6[13], YB6[13], YAT6[13], YBT6[13],
      YT6[13];  // Y6[-6]到Y6[6]变为Y6[0]到Y6[12], Y6=YA6+YB6*i ,即y=a+bi;

  for (int i = 0; i < NUM_PARTICLE; i++) {
    for (int q = 0; q < 13; q++) {
      YA6[q] = YB6[q] = YAT6[q] = YBT6[q] = YT6[q] = 0.0;
    }

    for (int j = 0; j < Maxlocal; j++) {
      int m = neigh_id[i][j];
      Vector<3> vtemp = pair_dv[i][m];

      cita = acos(vtemp[2] / vtemp.Norm());
      double tmp1 = cos(cita);
      double tmp2 = sin(cita);
      fai = atan2(vtemp[1], vtemp[0]);

      YA6[0] = cos(6 * fai) * pow(tmp2, 6) * sqrt(3003 / PI) / 64;
      YB6[0] = -sin(6 * fai) * pow(tmp2, 6) * sqrt(3003 / PI) / 64;
      YA6[1] = 3 * cos(5 * fai) * pow(tmp2, 5) * tmp1 * sqrt(1001 / PI) / 32;
      YB6[1] = -3 * sin(5 * fai) * pow(tmp2, 5) * tmp1 * sqrt(1001 / PI) / 32;
      YA6[2] = 3 * cos(4 * fai) * pow(tmp2, 4) * (11 * pow(tmp1, 2) - 1) *
               sqrt(91 / PI / 2) / 32;
      YB6[2] = -3 * sin(4 * fai) * pow(tmp2, 4) * (11 * pow(tmp1, 2) - 1) *
               sqrt(91 / PI / 2) / 32;
      YA6[3] = cos(3 * fai) * pow(tmp2, 3) * (11 * pow(tmp1, 3) - 3 * tmp1) *
               sqrt(1365 / PI) / 32;
      YB6[3] = -sin(3 * fai) * pow(tmp2, 3) * (11 * pow(tmp1, 3) - 3 * tmp1) *
               sqrt(1365 / PI) / 32;
      YA6[4] = cos(2 * fai) * pow(tmp2, 2) *
               (33 * pow(tmp1, 4) - 18 * pow(tmp1, 2) + 1) * sqrt(1365 / PI) /
               64;
      YB6[4] = -sin(2 * fai) * pow(tmp2, 2) *
               (33 * pow(tmp1, 4) - 18 * pow(tmp1, 2) + 1) * sqrt(1365 / PI) /
               64;
      YA6[5] = cos(fai) * tmp2 *
               (33 * pow(tmp1, 5) - 30 * pow(tmp1, 3) + 5 * tmp1) *
               sqrt(273 / PI / 2) / 16;
      YB6[5] = -sin(fai) * tmp2 *
               (33 * pow(tmp1, 5) - 30 * pow(tmp1, 3) + 5 * tmp1) *
               sqrt(273 / PI / 2) / 16;
      YA6[6] =
          (231 * pow(tmp1, 6) - 315 * pow(tmp1, 4) + 105 * pow(tmp1, 2) - 5) *
          sqrt(13 / PI) / 32;
      YB6[6] = 0;
      YA6[7] = -cos(fai) * tmp2 *
               (33 * pow(tmp1, 5) - 30 * pow(tmp1, 3) + 5 * tmp1) *
               sqrt(273 / 2 / PI) / 16;
      YB6[7] = -sin(fai) * tmp2 *
               (33 * pow(tmp1, 5) - 30 * pow(tmp1, 3) + 5 * tmp1) *
               sqrt(273 / 2 / PI) / 16;
      YA6[8] = cos(2 * fai) * pow(tmp2, 2) *
               (33 * pow(tmp1, 4) - 18 * pow(tmp1, 2) + 1) * sqrt(1365 / PI) /
               64;
      YB6[8] = sin(2 * fai) * pow(tmp2, 2) *
               (33 * pow(tmp1, 4) - 18 * pow(tmp1, 2) + 1) * sqrt(1365 / PI) /
               64;
      YA6[9] = -cos(3 * fai) * pow(tmp2, 3) * (11 * pow(tmp1, 3) - 3 * tmp1) *
               sqrt(1365 / PI) / 32;
      YB6[9] = -sin(3 * fai) * pow(tmp2, 3) * (11 * pow(tmp1, 3) - 3 * tmp1) *
               sqrt(1365 / PI) / 32;
      YA6[10] = 3 * cos(4 * fai) * pow(tmp2, 4) * (11 * pow(tmp1, 2) - 1) *
                sqrt(91 / PI / 2) / 32;
      YB6[10] = 3 * sin(4 * fai) * pow(tmp2, 4) * (11 * pow(tmp1, 2) - 1) *
                sqrt(91 / PI / 2) / 32;
      YA6[11] = -3 * cos(5 * fai) * pow(tmp2, 5) * tmp1 * sqrt(1001 / PI) / 32;
      YB6[11] = -3 * sin(5 * fai) * pow(tmp2, 5) * tmp1 * sqrt(1001 / PI) / 32;
      YA6[12] = cos(6 * fai) * pow(tmp2, 6) * sqrt(3003 / PI) / 64;
      YB6[12] = sin(6 * fai) * pow(tmp2, 6) * sqrt(3003 / PI) / 64;

      for (int t = 0; t < 13; t++) {
        YAT6[t] += YA6[t];
        YBT6[t] += YB6[t];
      }
    }

    for (int u = 0; u < 13; u++)
      YT6[u] = (YAT6[u] * YAT6[u] + YBT6[u] * YBT6[u]) / pow(Nb, 2.0);
    double temp = 0.0;
    for (int u = 0; u < 13; u++) temp += YT6[u];
    Y6 += sqrt(4.0 * PI * temp / 13.0);
  }

  return Y6 / double(NUM_PARTICLE);
}

// double CalculateQ6_rcut() {
//   int i, j;
//   CNode *pn0, *pn1;
//   CSuperball *ps0, *ps1;

//   int Nb = 0;
//   double cita, fai, Y6 = 0.0;
//   double YA6[13], YB6[13], YAT6[13], YBT6[13],
//       YT6[13];  // Y6[-6]到Y6[6]变为Y6[0]到Y6[12], Y6=YA6+YB6*i ,即y=a+bi;

//   double rcut = 2.008 * particles[0]->r_scale[0];

//   for (int q = 0; q < 13; q++) {
//     YA6[q] = YB6[q] = YAT6[q] = YBT6[q] = YT6[q] = 0.0;
//   }

//   for (int t = 0; t < NNN; t++) {
//     for (int m = 0; m < 14; m++) {
//       i = MASK3D[m][0], j = MASK3D[m][1];
//       for (pn0 = blocks[t]->head[i]; pn0 != NULL; pn0 = pn0->next) {
//         ps0 = pn0->ps;
//         if (pn0->PBC != NULL) {
//           psp[0]->Copyfrom(ps0);
//           psp[0]->Jump(pn0->PBC);
//           ps0 = psp[0];
//         }
//         for ((j == 0) ? (pn1 = pn0->next) : (pn1 = blocks[t]->head[j]);
//              pn1 != NULL; pn1 = pn1->next) {
//           ps1 = pn1->ps;
//           if (pn1->PBC != NULL) {
//             psp[1]->Copyfrom(ps1);
//             psp[1]->Jump(pn1->PBC);
//             ps1 = psp[1];
//           }
//           Vector<3> vtemp = ps0->center - ps1->center;
//           if (vtemp.Norm() < rcut) {
//             Nb++;
//             cita = acos(vtemp[2] / vtemp.Norm());
//             double tmp1 = cos(cita);
//             double tmp2 = sin(cita);
//             fai = atan2(vtemp[1], vtemp[0]);

//             YA6[0] = cos(6 * fai) * pow(tmp2, 6) * sqrt(3003 / PI) / 64;
//             YB6[0] = -sin(6 * fai) * pow(tmp2, 6) * sqrt(3003 / PI) / 64;
//             YA6[1] =
//                 3 * cos(5 * fai) * pow(tmp2, 5) * tmp1 * sqrt(1001 / PI) /
//                 32;
//             YB6[1] =
//                 -3 * sin(5 * fai) * pow(tmp2, 5) * tmp1 * sqrt(1001 / PI) /
//                 32;
//             YA6[2] = 3 * cos(4 * fai) * pow(tmp2, 4) * (11 * pow(tmp1, 2) -
//             1) *
//                      sqrt(91 / PI / 2) / 32;
//             YB6[2] = -3 * sin(4 * fai) * pow(tmp2, 4) *
//                      (11 * pow(tmp1, 2) - 1) * sqrt(91 / PI / 2) / 32;
//             YA6[3] = cos(3 * fai) * pow(tmp2, 3) *
//                      (11 * pow(tmp1, 3) - 3 * tmp1) * sqrt(1365 / PI) / 32;
//             YB6[3] = -sin(3 * fai) * pow(tmp2, 3) *
//                      (11 * pow(tmp1, 3) - 3 * tmp1) * sqrt(1365 / PI) / 32;
//             YA6[4] = cos(2 * fai) * pow(tmp2, 2) *
//                      (33 * pow(tmp1, 4) - 18 * pow(tmp1, 2) + 1) *
//                      sqrt(1365 / PI) / 64;
//             YB6[4] = -sin(2 * fai) * pow(tmp2, 2) *
//                      (33 * pow(tmp1, 4) - 18 * pow(tmp1, 2) + 1) *
//                      sqrt(1365 / PI) / 64;
//             YA6[5] = cos(fai) * tmp2 *
//                      (33 * pow(tmp1, 5) - 30 * pow(tmp1, 3) + 5 * tmp1) *
//                      sqrt(273 / PI / 2) / 16;
//             YB6[5] = -sin(fai) * tmp2 *
//                      (33 * pow(tmp1, 5) - 30 * pow(tmp1, 3) + 5 * tmp1) *
//                      sqrt(273 / PI / 2) / 16;
//             YA6[6] = (231 * pow(tmp1, 6) - 315 * pow(tmp1, 4) +
//                       105 * pow(tmp1, 2) - 5) *
//                      sqrt(13 / PI) / 32;
//             YB6[6] = 0;
//             YA6[7] = -cos(fai) * tmp2 *
//                      (33 * pow(tmp1, 5) - 30 * pow(tmp1, 3) + 5 * tmp1) *
//                      sqrt(273 / 2 / PI) / 16;
//             YB6[7] = -sin(fai) * tmp2 *
//                      (33 * pow(tmp1, 5) - 30 * pow(tmp1, 3) + 5 * tmp1) *
//                      sqrt(273 / 2 / PI) / 16;
//             YA6[8] = cos(2 * fai) * pow(tmp2, 2) *
//                      (33 * pow(tmp1, 4) - 18 * pow(tmp1, 2) + 1) *
//                      sqrt(1365 / PI) / 64;
//             YB6[8] = sin(2 * fai) * pow(tmp2, 2) *
//                      (33 * pow(tmp1, 4) - 18 * pow(tmp1, 2) + 1) *
//                      sqrt(1365 / PI) / 64;
//             YA6[9] = -cos(3 * fai) * pow(tmp2, 3) *
//                      (11 * pow(tmp1, 3) - 3 * tmp1) * sqrt(1365 / PI) / 32;
//             YB6[9] = -sin(3 * fai) * pow(tmp2, 3) *
//                      (11 * pow(tmp1, 3) - 3 * tmp1) * sqrt(1365 / PI) / 32;
//             YA6[10] = 3 * cos(4 * fai) * pow(tmp2, 4) *
//                       (11 * pow(tmp1, 2) - 1) * sqrt(91 / PI / 2) / 32;
//             YB6[10] = 3 * sin(4 * fai) * pow(tmp2, 4) *
//                       (11 * pow(tmp1, 2) - 1) * sqrt(91 / PI / 2) / 32;
//             YA6[11] =
//                 -3 * cos(5 * fai) * pow(tmp2, 5) * tmp1 * sqrt(1001 / PI) /
//                 32;
//             YB6[11] =
//                 -3 * sin(5 * fai) * pow(tmp2, 5) * tmp1 * sqrt(1001 / PI) /
//                 32;
//             YA6[12] = cos(6 * fai) * pow(tmp2, 6) * sqrt(3003 / PI) / 64;
//             YB6[12] = sin(6 * fai) * pow(tmp2, 6) * sqrt(3003 / PI) / 64;

//             for (int t = 0; t < 13; t++) {
//               YAT6[t] += YA6[t];
//               YBT6[t] += YB6[t];
//             }
//           }
//         }
//       }
//     }
//   }
//   if (Nb != 0) {
//     for (int s = 0; s < 13; s++) {
//       YAT6[s] /= double(Nb);
//       YBT6[s] /= double(Nb);
//     }
//   }

//   for (int u = 0; u < 13; u++) YT6[u] = YAT6[u] * YAT6[u] + YBT6[u] *
//   YBT6[u]; for (int v = 0; v < 13; v++) Y6 += YT6[v];

//   return sqrt(4 * PI * Y6 / 13);
// }

// double CalculateQ6local()
// {
// 	int i, j, ii, jj, m;
// 	double t, q6[Maxlocal], Q6ev[Maxlocal], Q6av;
// 	double x1, x2, x3, x4, x5, x6, y1, y2, y3, y4, y5, y6, z1, z2, z4, r2,
// r4, r6ni; 	double
// AB[Maxlocal][13];//开头为0的实部，0的虚部为0；之后从1到4的实部和虚部；
// 	CVector dcenter;

// 	for (i = 0; i < Maxlocal; i++)
// 		for (j = 0; j < 13; j++)
// 			AB[i][j] = 0.0;
// 	for (ii = 0; ii < Npolyhedron; ii++)
// 	{
// 		for (jj = 0; jj < Maxlocal; jj++)
// 		{
// 			m = pairnumbermin[ii][jj];
// 			//if (m == ii) { cout << "error!! 颗粒本身统计" << endl;
// getchar(); } 			dcenter = pairdcenter[ii][m];
// x1 = dcenter[0]; y1 = dcenter[1]; z1 = dcenter[2]; 			x2 =
// x1*x1; y2 = y1*y1; z2 = z1*z1; 			x3 = x2*x1; y3 = y2*y1;
// 			x4 = x3*x1; y4 = y3*y1; z4 = z2*z2;
// 			x5 = x4*x1; y5 = y4*y1;
// 			x6 = x5*x1; y6 = y5*y1;
// 			r2 = x2 + y2 + z2; r4 = r2*r2; r6ni = 1.0 / (r2*r4);

// 			//m=0;
// 			t = (231.0*z2*z4 - 315.0*z4*r2 + 105.0*z2*r4)*r6ni
// - 5.0; 			AB[jj][0] = AB[jj][0] + t;
// 			//m=1;
// 			t = z1*(33.0*z4 - 30.0*z2*r2 + 5.0*r4)*r6ni;
// 			AB[jj][1] = AB[jj][1] + t*x1;
// 			AB[jj][2] = AB[jj][2] + t*y1;
// 			//m=2;
// 			t = (33.0*z4 - 18.0*z2*r2 + r4)*r6ni;
// 			AB[jj][3] = AB[jj][3] + t*(x2 - y2);
// 			AB[jj][4] = AB[jj][4] + t*2.0*x1*y1;
// 			//m=3;
// 			t = z1*(11.0*z2 - 3.0*r2)*r6ni;
// 			AB[jj][5] = AB[jj][5] + t*(x3 - 3.0*x1*y2);
// 			AB[jj][6] = AB[jj][6] + t*(3.0*x2*y1 - y3);
// 			//m=4;
// 			t = (11.0*z2 - r2) * r6ni;
// 			AB[jj][7] = AB[jj][7] + t*(x4 - 6.0*x2*y2 + y4);
// 			AB[jj][8] = AB[jj][8] + t*4.0*(x3*y1 - x1*y3);
// 			//m=5;
// 			t = z1 * r6ni;
// 			AB[jj][9] = AB[jj][9] + t*(x5 - 10.0*x3*y2 + 5.0*x1*y4);
// 			AB[jj][10] = AB[jj][10] + t*(5.0*x4*y1 - 10.0*x2*y3 +
// y5);
// 			//m=6;
// 			t = r6ni;
// 			AB[jj][11] = AB[jj][11] + t*(x6 - 15.0*x4*y2
// + 15.0*x2*y4
// -
// y6); 			AB[jj][12] = AB[jj][12] + t*(6.0*x5*y1
// - 20.0*x3*y3
// + 6.0*x1*y5);
// 		}
// 	}
// 	for (i = 1; i < Maxlocal; i++)
// 		for (j = 0; j < 13; j++)
// 			AB[i][j] = AB[i][j] + AB[i - 1][j];

// 	////2范数
// 	//Q6av = 0.0;
// 	//for (i = 0; i < Maxlocal; i++)
// 	//{
// 	//	q6[i] = 4.0*AB[i][0] * AB[i][0];
// 	//	q6[i] = q6[i] + 336.0*(AB[i][1] * AB[i][1] + AB[i][2] *
// AB[i][2]);
// 	//	q6[i] = q6[i] + 210.0*(AB[i][3] * AB[i][3] + AB[i][4] *
// AB[i][4]);
// 	//	q6[i] = q6[i] + 840.0*(AB[i][5] * AB[i][5] + AB[i][6] *
// AB[i][6]);
// 	//	q6[i] = q6[i] + 252.0*(AB[i][7] * AB[i][7] + AB[i][8] *
// AB[i][8]);
// 	//	q6[i] = q6[i] + 5544.0*(AB[i][9] * AB[i][9] + AB[i][10] *
// AB[i][10]);
// 	//	q6[i] = q6[i] + 462.0*(AB[i][11] * AB[i][11] + AB[i][12] *
// AB[i][12]);
// 	//	q6[i] = sqrt(q6[i]);
// 	//	Q6ev[i] = q6[i] / (32.0*double(Npolyhedron*(i + 1)));
// 	//	Q6ev[i] = fabs((Q6ev[i] - Q6random[i]) / Q6length[i]);
// 	//	Q6av = Q6av + Q6ev[i] * Q6ev[i];
// 	//}
// 	//Q6av = sqrt(Q6av / double(Maxlocal));

// 	//无穷范数
// 	Q6av = 0.0;
// 	for (i = 0; i < Maxlocal; i++)
// 	{
// 		q6[i] = 4.0*AB[i][0] * AB[i][0];
// 		q6[i] = q6[i] + 336.0*(AB[i][1] * AB[i][1] + AB[i][2] *
// AB[i][2]); 		q6[i] = q6[i] + 210.0*(AB[i][3] * AB[i][3] + AB[i][4] *
// AB[i][4]); 		q6[i] = q6[i] + 840.0*(AB[i][5] * AB[i][5] + AB[i][6] *
// AB[i][6]); q6[i] = q6[i] + 252.0*(AB[i][7] * AB[i][7] + AB[i][8] * AB[i][8]);
// q6[i] = q6[i] + 5544.0*(AB[i][9] * AB[i][9] + AB[i][10] * AB[i][10]);
// q6[i] = q6[i] + 462.0*(AB[i][11] * AB[i][11] + AB[i][12]
// * AB[i][12]); 		q6[i] = sqrt(q6[i]); 		Q6ev[i] = q6[i]
// / (32.0*double(Npolyhedron*(i + 1))); 		Q6ev[i] = fabs((Q6ev[i]
// -
// Q6localrandom[i]) / Q6locallength[i]); 		if (Q6av < Q6ev[i]) Q6av
// = Q6ev[i];
// 	}
// 	result << Q6av << "\t";
// 	return Q6av;
// }

void Sph_harmonics(int l, int m, Vector<3> r) {
  double A, P, e;

  int start = 1;
  for (int i = (l - m + 1); i <= (l + m); i++) {
    start *= i;
  }
  A = sqrt((2 * l + 1) / 4.0 / PI / double(start));
}

// // ----------------
// // Hyperuniform Analysis
// // ----------------
// void Structure_factor(int species, bool global = true) {
//   int nk = int(L / sqrt(3.0));  // a rough estimate
//   double thickness = 2.0 * PI / L;

//   double Sf[nk];
//   int num[nk];

//   for (int i = 0; i < nk; ++i) {
//     Sf[i] = 0.0;
//     num[i] = 0;
//   }

//   double Re, Im;

//   for (int i = -nk; i < nk; ++i) {
//     for (int j = -nk; j < nk; ++j) {
//       for (int k = -nk; k < nk; ++k) {
//         CVector K;
//         K.Set(thickness * i, thickness * j, thickness * k);

//         Re = Im = 0.0;
//         double sf;
//         if (global) {
//           for (int n = 0; n < NUM_PARTICLE; ++n) {
//             double temp = K.Dot(particles[n]->center);
//             Re += cos(temp);
//             Im += sin(temp);
//           }
//           sf = (Re * Re + Im * Im) / NUM_PARTICLE;
//         } else {
//           int id = 0;
//           for (int i = 0; i < species - 1; ++i) id += NpEach[i];

//           for (int n = 0; n < NpEach[species]; ++n) {
//             double temp = K.Dot(particles[id + n]->center);
//             Re += cos(temp);
//             Im += sin(temp);
//           }
//           sf = (Re * Re + Im * Im) / NpEach[species];
//         }

//         int index = int(K.Length() / thickness);
//         Sf[index] += sf;
//         num[index] += 1;
//       }
//     }
//   }

//   ofstream output("SF.txt");
//   output << "species\t" << species << endl;
//   output << global << endl;
//   for (int i = 0; i < nk; ++i) {
//     if (num[i] == 0) continue;
//     Sf[i] /= num[i];
//     output << (i + 0.5) * thickness << "\t" << Sf[i] << endl;
//   }
//   output.close();
// }

// void spectral_density(int species, bool global = true) {
//   int nk = int(L / sqrt(3.0));  // a rough estimate
//   double thickness = 2.0 * PI / L;
//   double density[nk];

//   for (int i = 0; i < nk; ++i) density[i] = 0;

//   // _m: the Fourier transform of the indicator function
//   double Re, Im, Re_m, Im_m;
//   for (int i = -nk; i < nk; ++i) {
//     for (int j = -nk; j < nk; ++j) {
//       for (int k = -nk; k < nk; ++k) {
//         CVector K;
//         K.Set(thickness * i, thickness * j, thickness * k);

//         Re = Im = 0.0;
//         if (global) {
//           for (int n = 0; n < NUM_PARTICLE; ++n) {
//             double temp = -K.Dot(particles[n]->center);
//             FFT_if(particles[n], 0.01, K, &Re_m, &Im_m);

//             Re += cos(temp) * Re_m - sin(temp) * Im_m;
//             Im += cos(temp) * Im_m + sin(temp) * Re_m;
//           }
//         } else {
//           int id = 0;
//           for (int i = 0; i < species - 1; ++i) id += NpEach[i];

//           for (int n = 0; n < NpEach[species]; ++n) {
//             double temp = -K.Dot(particles[id + n]->center);
//             FFT_if(particles[id + n], 0.01, K, &Re_m, &Im_m);
//             cout << n << endl;

//             Re += cos(temp) * Re_m - sin(temp) * Im_m;
//             Im += cos(temp) * Im_m + sin(temp) * Re_m;
//           }
//         }

//         double sd = (Re * Re + Im * Im) / pow(L, 3.0);
//         int index = int(K.Length() / thickness);

//         density[index] += sd;
//       }
//     }
//   }

//   ofstream output("SD.txt");
//   output << "species\t" << species << endl;
//   output << global << endl;
//   for (int i = 0; i < nk; ++i) {
//     double Vshell =
//         4.0 / 3.0 * PI * (3 * i * i + 3 * i + 1) * pow(thickness, 3.0);
//     density[i] /= Vshell;

//     output << (i + 0.5) * thickness << "\t" << density[i] << endl;
//   }

//   output.close();
// }

// // ----------------
// // Voronoi Analysis
// // ----------------
// // Discretize surface for a given species under local coordinates
// void getsurpoint(double para[], int num, CVector *SurfPot) {
//   // Discretize a superellipsoidal particle surface along the longitude
//   // direction into 20 (Yuan: NvA(s1,s2)) equal parts (40 (2NvA(s1,s2)) parts
//   // for the latitude direction, Nv = 12)
//   double p = para[0];

//   int id;
//   double dv = PI / (double)num;

//   int NSuP = 2 * (num * num - num + 1);  // 40*20-20*2+2
//   // SurfPot = new CVector[NSuP];

//   double *sin_fai, *cos_fai, *sin_cita, *cos_cita;
//   sin_cita = new double[num - 1];  // avoid replication
//   cos_cita = new double[num - 1];
//   sin_fai = new double[2 * num];
//   cos_fai = new double[2 * num];

//   // sin(cita)cos(fai), sin(cita)sin(fai), cos(cita)

//   // v=cita (0, PI)
//   // start from 1 to avoid replication
//   for (int i = 0; i < num - 1; ++i) {
//     double v = dv * double(i + 1);
//     sin_cita[i] = sin(v);
//     cos_cita[i] = cos(v);
//   }
//   // u=fai [0, 2*PI)
//   for (int j = 0; j < 2 * num; ++j) {
//     double u = dv * double(j);
//     sin_fai[j] = sin(u);
//     cos_fai[j] = cos(u);
//   }
//   SurfPot[0].Set(0.0, 0.0, para[3]);

//   for (int i = 0; i < num - 1; ++i) {
//     for (int j = 0; j < 2 * num; ++j) {
//       id = i * 2 * num + j + 1;
//       SurfPot[id].Set(para[1] * pow(sin_cita[i] * cos_fai[j], 1.0 / p),
//                       para[2] * pow(sin_cita[i] * sin_fai[j], 1.0 / p),
//                       para[3] * pow(cos_cita[i], 1.0 / p));
//     }
//   }

//   id += 1;
//   SurfPot[id].Set(0.0, 0.0, -para[3]);

//   delete[] sin_cita;
//   delete[] cos_cita;
//   delete[] sin_fai;
//   delete[] cos_fai;
// }

// // num: Precision
// void OutputVoroPoint(int num, int replica) {
//   char cha[100];

//   sprintf(cha, "../vorofile/Surfpoint%d.txt", replica);
//   ofstream scr(cha);
//   scr << fixed << setprecision(10);

//   CVector center, point_t, point;
//   double A[3][3];

//   int NSuP = 2 * (num * num - num + 1);  // 40*20-20*2+2

//   int id = -1;
//   for (int i = 0; i < Nkind; ++i) {
//     CVector *SurfPot;
//     SurfPot = new CVector[NSuP];
//     getsurpoint(Para[i], num, SurfPot);

//     for (int j = 0; j < NpEach[i]; ++j) {
//       id++;
//       center = particles[id]->center;
//       for (int m = 0; m < 3; ++m)
//         for (int n = 0; n < 3; ++n) {
//           A[n][m] = particles[id]->e[m][n];
//         }
//       for (int t = 0; t < NSuP; ++t) {
//         point_t = SurfPot[t];  // temp
//         MatrixMult(A, point_t, point);
//         point += center;
//         PeriodicCheck(point);
//         int index = id * NSuP + t;
//         scr << index << "\t" << point[0] << "\t" << point[1] << "\t" <<
//         point[2]
//             << endl;
//       }
//     }

//     delete[] SurfPot;
//   }
//   scr.close();
// }

// void OutputBoundary(string filename, int replica) {
//   ofstream output;
//   output.open(filename, ios::app);

//   output << replica << "\t" << L << endl;

//   output.close();
// }

// void VoroVolume(int num, int replica) {
//   char cha[100];
//   sprintf(cha, "../vorofile/V_voro%d.txt", replica);
//   ifstream voro(cha);
//   if (!voro.good()) {
//     cout << "Unable to open file!" << endl;
//     exit(1);
//   }

//   sprintf(cha, "../vorofile/V_analysis%d.txt", replica);
//   ofstream scr(cha);
//   scr << fixed << setprecision(15);

//   double *Vlocal, *PDlocal;
//   double *ave_v, *ave_pd, *sigma_v, *sigma_pd;

//   Vlocal = new double[NUM_PARTICLE];
//   PDlocal = new double[NUM_PARTICLE];

//   ave_v = new double[Nkind + 1];
//   ave_pd = new double[Nkind + 1];
//   sigma_v = new double[Nkind + 1];
//   sigma_pd = new double[Nkind + 1];

//   int NSuP = 2 * (num * num - num + 1);

//   for (int i = 0; i < NUM_PARTICLE; ++i) Vlocal[i] = 0.0;

//   int n, id;
//   double x, y, z, v;
//   voro >> n >> x >> y >> z >> v;

//   while (1) {
//     if (voro.eof()) break;
//     // scr << n << "\t" << x << "\t" << y << "\t" << z << "\t" << v << endl;
//     id = int(n / NSuP);
//     Vlocal[id] += v;
//     voro >> n >> x >> y >> z >> v;
//   }

//   scr << "index\tkind\tVorov\tPDlocal" << endl;
//   id = -1;
//   for (int i = 0; i < Nkind; ++i) {
//     for (int j = 0; j < NpEach[i]; ++j) {
//       id++;
//       Vlocal[id] /= particles[id]->vol;
//       PDlocal[id] = 1.0 / Vlocal[id];
//       scr << id << "\t" << i << "\t" << Vlocal[id] << "\t" << PDlocal[id]
//           << endl;
//     }
//   }

//   id = -1;
//   ave_v[Nkind] = ave_pd[Nkind] = 0.0;
//   for (int i = 0; i < Nkind; ++i) {
//     ave_v[i] = ave_pd[i] = 0.0;
//     for (int j = 0; j < NpEach[i]; ++j) {
//       id++;
//       ave_v[i] += Vlocal[id];
//       ave_pd[i] += PDlocal[id];
//     }
//     ave_v[Nkind] += ave_v[i];
//     ave_pd[Nkind] += ave_pd[i];

//     if (NpEach[i] > 0) {
//       ave_v[i] /= double(NpEach[i]);
//       ave_pd[i] /= double(NpEach[i]);
//     }
//   }
//   ave_v[Nkind] /= double(NUM_PARTICLE);
//   ave_pd[Nkind] /= double(NUM_PARTICLE);

//   id = -1;
//   sigma_v[Nkind] = sigma_pd[Nkind] = 0.0;
//   for (int i = 0; i < Nkind; ++i) {
//     sigma_v[i] = sigma_pd[i] = 0.0;
//     for (int j = 0; j < NpEach[i]; ++j) {
//       id++;
//       sigma_v[Nkind] += pow(Vlocal[id] - ave_v[Nkind], 2.0);
//       sigma_pd[Nkind] += pow(PDlocal[id] - ave_pd[Nkind], 2.0);
//       sigma_v[i] += pow(Vlocal[id] - ave_v[i], 2.0);
//       sigma_pd[i] += pow(PDlocal[id] - ave_pd[i], 2.0);
//     }
//     if (NpEach[i] > 0) {
//       sigma_v[i] = sqrt(sigma_v[i] / double(NpEach[i] - 1));
//       sigma_pd[i] = sqrt(sigma_pd[i] / double(NpEach[i] - 1));
//     }
//   }
//   sigma_v[Nkind] = sqrt(sigma_v[Nkind] / double(NUM_PARTICLE - 1));
//   sigma_pd[Nkind] = sqrt(sigma_pd[Nkind] / double(NUM_PARTICLE - 1));

//   scr << "Sum:" << endl;
//   scr << ave_v[Nkind] << "\t" << sigma_v[Nkind] << "\t" << ave_pd[Nkind] <<
//   "\t"
//       << sigma_pd[Nkind] << endl;

//   for (int i = 0; i < Nkind; ++i) {
//     scr << NpEach[i] << "\t" << ave_v[i] << "\t" << sigma_v[i] << "\t"
//         << ave_pd[i] << "\t" << sigma_pd[i] << endl;
//   }

//   voro.close();
//   scr.close();
//   delete[] Vlocal;
//   delete[] PDlocal;
//   delete[] ave_v;
//   delete[] ave_pd;
//   delete[] sigma_v;
//   delete[] sigma_pd;
// }

// ----------------
// Other utils
// ----------------
void PeriodicCheck(Vector<3> &point) {
  for (int i = 0; i < 3; ++i) {
    while (point[i] >= L) {
      point[i] -= L;
    }
    while (point[i] < 0) {
      point[i] += L;
    }
  }
}

// ----------------------------------------------------------------
// Background grid
// ----------------------------------------------------------------
void BackGroundGrids() {
  int i, j;
  double X[2], Y[2], Z[2];
  CSuperball *ps;
  if (!blocks.empty()) {
    for (i = 0; i < blocks.size(); i++) delete blocks[i];
    blocks.clear();
  }
  l = particles[0]->r_scale[0];
  nb = int(Lambda[0][0] / l);
  if (nb < 1) nb = 1;
  l = L / double(nb);
  NNN = nb * nb * nb;
  for (i = 0; i < NNN; i++) blocks.push_back(new CCell());
  for (i = 0; i < NUM_PARTICLE; i++) {
    ps = particles[i];
    ps->Boundary(X, Y, Z);
    for (j = 0; j < 2; j++) {
      ps->ix[j] = int(X[j] / l);
      ps->iy[j] = int(Y[j] / l);
      ps->iz[j] = int(Z[j] / l);
    }
    if (X[0] < 0.0) ps->ix[0]--;
    if (Y[0] < 0.0) ps->iy[0]--;
    if (Z[0] < 0.0) ps->iz[0]--;
    AddToBlocks(ps);
  }
}

void AddToBlocks(CSuperball *ps) {
  // need to compute ps->ix[2],iy[2],iz[2] first
  int j, k, ii, jj, kk, iii, jjj, kkk, nnn, p;
  double *PBC;
  CNode *pn;

  ps->dix = ps->ix[1] - ps->ix[0] + 1;
  ps->diy = ps->iy[1] - ps->iy[0] + 1;
  ps->diz = ps->iz[1] - ps->iz[0] + 1;
  ps->nob = ps->dix * ps->diy * ps->diz;
  if (ps->bl != NULL) delete[] ps->bl;
  if (ps->hl != NULL) delete[] ps->hl;
  if (ps->nl != NULL) delete[] ps->nl;
  ps->bl = new int[ps->nob];
  ps->hl = new int[ps->nob];
  ps->nl = new CNode *[ps->nob];

  // 颗粒在一个网格内
  if (ps->nob == 1) {
    {
      k = ps->iz[0] * nb * nb + ps->iy[0] * nb + ps->ix[0];
      j = 0;
      PBC = NULL;
      pn = new CNode(ps, PBC);
      blocks[k]->Add(pn, j);
      ps->bl[0] = k;
      ps->hl[0] = j;
      ps->nl[0] = pn;
    }
    return;
  }
  // 颗粒不跨边界 跨网格
  if ((ps->ix[0] > -1) && (ps->ix[1] < nb) && (ps->iy[0] > -1) &&
      (ps->iy[1] < nb) && (ps->iz[0] > -1) && (ps->iz[1] < nb)) {
    for (kk = ps->iz[0]; kk <= ps->iz[1]; kk++) {
      for (jj = ps->iy[0]; jj <= ps->iy[1]; jj++) {
        for (ii = ps->ix[0]; ii <= ps->ix[1]; ii++) {
          k = kk * nb * nb + jj * nb + ii;  // 网格号
          j = 0;                            // 分类号
          if (ii > ps->ix[0]) j = (j | 1);
          if (jj > ps->iy[0]) j = (j | 2);
          if (kk > ps->iz[0]) j = (j | 4);
          PBC = NULL;
          pn = new CNode(ps, PBC);
          blocks[k]->Add(pn, j);
          nnn = (kk - ps->iz[0]) * ps->diy * ps->dix +
                (jj - ps->iy[0]) * ps->dix + (ii - ps->ix[0]);
          ps->bl[nnn] = k;
          ps->hl[nnn] = j;
          ps->nl[nnn] = pn;
        }
      }
    }
    return;
  }
  // 一般情形
  for (kk = ps->iz[0]; kk <= ps->iz[1]; kk++) {
    for (jj = ps->iy[0]; jj <= ps->iy[1]; jj++) {
      for (ii = ps->ix[0]; ii <= ps->ix[1]; ii++) {
        j = 0;  // 分类号
        if (ii > ps->ix[0]) j = (j | 1);
        if (jj > ps->iy[0]) j = (j | 2);
        if (kk > ps->iz[0]) j = (j | 4);
        PBC = new double[3];

        p = 0;
        if (ii < 0)
          p = 1 - ((ii + 1) / nb);
        else
          p = -(ii / nb);
        iii = ii + p * nb;
        PBC[0] = double(p) * L;

        p = 0;
        if (jj < 0)
          p = 1 - ((jj + 1) / nb);
        else
          p = -(jj / nb);
        jjj = jj + p * nb;
        PBC[1] = double(p) * L;

        p = 0;
        if (kk < 0)
          p = 1 - ((kk + 1) / nb);
        else
          p = -(kk / nb);
        kkk = kk + p * nb;
        PBC[2] = double(p) * L;

        k = kkk * nb * nb + jjj * nb + iii;
        if ((ii == iii) && (jj == jjj) && (kk == kkk)) {
          delete[] PBC;
          PBC = NULL;
        }
        pn = new CNode(ps, PBC);
        blocks[k]->Add(pn, j);
        nnn = (kk - ps->iz[0]) * ps->diy * ps->dix +
              (jj - ps->iy[0]) * ps->dix + (ii - ps->ix[0]);
        ps->bl[nnn] = k;
        ps->hl[nnn] = j;
        ps->nl[nnn] = pn;
      }
    }
  }
}
void DeleteNode(int blocknumber, int headnumber, CNode *pn) {
  if (pn->next != NULL) pn->next->prev = pn->prev;
  if (pn->prev != NULL)
    pn->prev->next = pn->next;
  else
    blocks[blocknumber]->head[headnumber] = pn->next;
  delete pn;
}

// ----------------------------------------------------------------
// File output
// ----------------------------------------------------------------
void POV_superball(string filename) {
  // green, blue
  double color[3][3] = {
      {0.484, 0.984, 0}, {0, 0.75, 1.0}, {0.574, 0.4375, 0.855}};
  // p < 1.0 red p>1.0 green
  double len = 1.8;

  Matrix<3, 3> cell = Matrix<3, 3>::UNIT();

  CSuperball *ps;
  ofstream pov(filename);
  pov << "#include \"colors.inc\"" << endl;
  pov << "#include \"textures.inc\"" << endl;
  pov << "camera{ location <" << len << "," << len << "," << len
      << "> look_at <0,0,0> }" << endl;
  pov << "background {rgb 1}" << endl;
  pov << "light_source{ <60,60,10> color rgb <1,1,1> }" << endl;
  pov << "light_source{ <10,60,60> color rgb <1,1,1> }" << endl;

  double br = 0.0025;
  Vector<3> p_sc;
  Vector<3> cs_, se_, cs, se;
  int bound_flag[3];
  cs = Vector<3>::ZERO();
  for (int i = 0; i < 3; i++) {
    bound_flag[0] = bound_flag[1] = bound_flag[2] = 0;
    bound_flag[i] = 1;
    se_ = {double(bound_flag[0]), double(bound_flag[1]), double(bound_flag[2])};
    se = cell * se_;

    pov << "cylinder { <0,0,0> , <" << se[0] << "," << se[1] << "," << se[2]
        << "> ," << br << " texture{ pigment{\t";
    pov << "color rgb <0.625,0.125,0.9375> } finish{ phong 1.0 reflection 0.0 "
           "} } }"
        << endl;  // purple
  }
  cs_ = Vector<3>::ZERO();
  cs = cell * cs_;
  for (int i = 0; i < 3; i++) {
    bound_flag[0] = bound_flag[1] = bound_flag[2] = 1;
    bound_flag[i] = 0;
    se_ = {double(bound_flag[0]), double(bound_flag[1]), double(bound_flag[2])};
    se = cell * se_;
    pov << "cylinder { <" << cs[0] << "," << cs[1] << "," << cs[2] << "> , <"
        << se[0] << "," << se[1] << "," << se[2] << "> ," << br
        << " texture{ pigment{\t";
    pov << "color rgb <0.625,0.125,0.9375> } finish{ phong 1.0 reflection 0.0 "
           "} } }"
        << endl;  // purple
  }
  for (int i = 0; i < 3; i++) {
    bound_flag[0] = bound_flag[1] = bound_flag[2] = 0;
    bound_flag[i] = 1;
    cs_ = {double(bound_flag[0]), double(bound_flag[1]), double(bound_flag[2])};
    cs = cell * cs_;
    for (int j = 0; j < 3; j++) {
      if (j == i) continue;
      bound_flag[0] = bound_flag[1] = bound_flag[2] = 1;
      bound_flag[j] = 0;
      se_ = {double(bound_flag[0]), double(bound_flag[1]),
             double(bound_flag[2])};
      se = cell * se_;
      pov << "cylinder { <" << cs[0] << "," << cs[1] << "," << cs[2] << "> , <"
          << se[0] << "," << se[1] << "," << se[2] << "> ," << br
          << " texture{ pigment{\t";
      pov << "color rgb <0.625,0.125,0.9375> } finish{ phong 1.0 reflection "
             "0.0 } } }"
          << endl;  // purple
    }
  }

  int id = -1;
  for (int m = 0; m < Nkind; ++m) {
    double c[3];
    c[0] = color[m][0];
    c[1] = color[m][1];
    c[2] = color[m][2];

    for (int i = 0; i < NpEach[m]; i++) {
      id++;
      ps = particles[id];

      pov << "object { superellipsoid { <" << 1.0 / ps->p << "," << 1.0 / ps->p
          << "> scale <" << ps->r_scale[0] << "," << ps->r_scale[1] << ","
          << ps->r_scale[2] << "> texture{ pigment{\t";
      // DW: 0.484,0.984,0; YY: 0.625,0.125,0.9375 (green)
      pov << "color rgb <" << c[0] << "," << c[1] << "," << c[2]
          << "> } finish{ phong 1.0 reflection 0.0 } } } matrix<";
      for (int j = 0; j < 3; j++)
        pov << ps->e[j][0] << "," << ps->e[j][1] << "," << ps->e[j][2] << ",";
      pov << ps->center[0] << "," << ps->center[1] << "," << ps->center[2]
          << "> }" << endl;
    }
  }
  pov.close();
}

void XYZ_output(string filename) {
  double color[3] = {0.498039215686, 0.498039215686, 0.513725490196};

  ofstream xyz(filename);

  xyz << NUM_PARTICLE << endl;
  xyz << "Lattice=\"" << Lambda[0][0] << ' ' << Lambda[0][1] << ' '
      << Lambda[0][2] << ' ' << Lambda[1][0] << ' ' << Lambda[1][1] << ' '
      << Lambda[1][2] << ' ' << Lambda[2][0] << ' ' << Lambda[2][1] << ' '
      << Lambda[2][2] << "\"" << ' ';
  xyz << "Properties=pos:R:3:aspherical_shape:R:3:color:R:3" << endl;

  for (int i = 0; i < NUM_PARTICLE; ++i) {
    xyz << fixed << setprecision(6) << particles[i]->center[0] << ' '
        << particles[i]->center[1] << ' ' << particles[i]->center[2] << ' '
        << particles[i]->r_scale[0] << ' ' << particles[i]->r_scale[1] << ' '
        << particles[i]->r_scale[2] << ' ' << color[0] << ' ' << color[1] << ' '
        << color[2] << endl;
  }
}