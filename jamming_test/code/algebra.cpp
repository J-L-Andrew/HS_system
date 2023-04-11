#include "algebra.h"
/************************* Vector Definition *************************/

template <int ROW>
Vector<ROW>::Vector() {
  vec_.fill(0);
}

template <>
Vector<2>::Vector(double x, double y) {
  vec_ = {x, y};
}

template <>
Vector<3>::Vector(double x, double y) {
  vec_ = {x, y, 0};
}

template <>
Vector<2>::Vector(double x, double y, double z) {
  vec_ = {x, y};
}

template <>
Vector<3>::Vector(double x, double y, double z) {
  vec_ = {x, y, z};
}

template <int ROW>
Vector<ROW>::Vector(std::array<double, ROW> vec) {
  vec_ = vec;
}

template <int ROW>
double& Vector<ROW>::operator[](int i) {
  return vec_[i];
}

template <int ROW>
const double& Vector<ROW>::operator[](int i) const {
  return vec_[i];
}

template <int ROW>
bool Vector<ROW>::operator==(const Vector& vec) const {
  for (int i = 0; i < ROW; ++i) {
    if (fabs(vec_[i] - vec[i]) < 1e-6) {
      return false;
    }
  }
  return true;
}

template <int ROW>
Vector<ROW> Vector<ROW>::operator-() const {
  Vector<ROW> res;
  for (int i = 0; i < ROW; ++i) {
    res[i] = -vec_[i];
  }
  return res;
}

template <int ROW>
Vector<ROW> Vector<ROW>::operator+(const Vector& vec) const {
  Vector<ROW> res;
  for (int i = 0; i < ROW; ++i) {
    res[i] = vec_[i] + vec[i];
  }
  return res;
}

template <int ROW>
Vector<ROW> Vector<ROW>::operator-(const Vector& vec) const {
  Vector<ROW> res;
  for (int i = 0; i < ROW; ++i) {
    res[i] = vec_[i] - vec[i];
  }
  return res;
}

template <int ROW>
Vector<ROW>& Vector<ROW>::operator+=(const Vector& vec) {
  for (int i = 0; i < ROW; ++i) {
    vec_[i] += vec[i];
  }
  return *this;
}

template <int ROW>
Vector<ROW>& Vector<ROW>::operator-=(const Vector& vec) {
  for (int i = 0; i < ROW; ++i) {
    vec_[i] -= vec[i];
  }
  return *this;
}

template <int ROW>
Vector<ROW> Vector<ROW>::operator*(double val) const {
  Vector<ROW> res;
  for (int i = 0; i < ROW; ++i) {
    res[i] = vec_[i] * val;
  }
  return res;
}

template <int ROW>
Vector<ROW> Vector<ROW>::operator/(double val) const {
  Vector<ROW> res;
  for (int i = 0; i < ROW; ++i) {
    res[i] = vec_[i] / val;
  }
  return res;
}

template <int ROW>
Vector<ROW>& Vector<ROW>::operator*=(double val) {
  for (int i = 0; i < ROW; ++i) {
    vec_[i] *= val;
  }
  return *this;
}

template <int ROW>
Vector<ROW>& Vector<ROW>::operator/=(double val) {
  for (int i = 0; i < ROW; ++i) {
    vec_[i] /= val;
  }
  return *this;
}

template <int ROW>
double Vector<ROW>::Dot(const Vector& vec) const {
  double res = 0;
  for (int i = 0; i < ROW; ++i) {
    res += vec_[i] * vec[i];
  }
  return res;
}

template <>
Vector<3> Vector<2>::Cross(const Vector<2>& vec) const {
  Vector<3> res;
  res[2] = vec_[0] * vec[1] - vec_[1] * vec[0];
  return res;
}

template <>
Vector<3> Vector<3>::Cross(const Vector<3>& vec) const {
  Vector<3> res;
  res[0] = vec_[1] * vec[2] - vec_[2] * vec[1];
  res[1] = vec_[2] * vec[0] - vec_[0] * vec[2];
  res[2] = vec_[0] * vec[1] - vec_[1] * vec[0];
  return res;
}

template <int ROW>
Matrix<ROW, ROW> Vector<ROW>::Dyadic(const Vector& vec) const {
  Matrix<ROW, ROW> res;
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < ROW; ++j) {
      res[i][j] = vec_[i] * vec[j];
    }
  }
  return res;
}

template <int ROW>
double Vector<ROW>::MaxElement() const {
  double max = vec_[0];
  for (int i = 1; i < ROW; i++) {
    if (max < vec_[i]) max = vec_[i];
  }
  return max;
}

template <int ROW>
double Vector<ROW>::Norm() const {
  return std::sqrt(this->Dot(*this));
}

template <int ROW>
Vector<2> Vector<ROW>::Normal() const {
  Vector<2> normal;
  normal[0] = -vec_[1];
  normal[1] = vec_[0];
  return normal;
}

template <int ROW>
Vector<ROW> Vector<ROW>::Normalize() const {
  double norm = 1.0 / this->Norm();
  if (norm == 0.) {
    return *this;
  } else {
    return (*this) * norm;
  }
}

template <int ROW>
Vector<ROW> Vector<ROW>::ZERO() {
  Vector<ROW> zero_vec;
  return zero_vec;
}
template <int ROW>
Vector<ROW> Vector<ROW>::UNIT() {
  Vector<ROW> unit_vec;
  for (int i = 0; i < ROW; i++) {
    unit_vec[i] = 1;
  }
  return unit_vec;
}

template <int ROW>
Vector<ROW> Vector<ROW>::RAND(double low, double up) {
  // srand(static_cast<int>(time(nullptr) + std::rand()));
  Vector<ROW> rand_vec;
  for (int i = 0; i < ROW; ++i) {
    rand_vec[i] = low + double(std::rand()) / (double(RAND_MAX) + 1) * (up - low);
  }
  return rand_vec;
}

template <int ROW>
Vector<ROW> Vector<ROW>::RAND(const Matrix<ROW, ROW>& mat) {
  // srand(static_cast<int>(time(nullptr) + std::rand()));
  Vector<ROW> rand_vec;
  for (int i = 0; i < ROW; ++i) {
    rand_vec += double(std::rand()) / (double(RAND_MAX) + 1) * Vector<ROW>(mat[i]);
  }
  return rand_vec;
}

Vector<2> operator*(const double& val, const Vector<2>& vec) { return vec * val; }
Vector<3> operator*(const double& val, const Vector<3>& vec) { return vec * val; }
std::ostream& operator<<(std::ostream& out, const Vector<2>& vec) {
  out << "[ ";
  for (int i = 0; i < 2; ++i) {
    out << vec[i] << "; ";
  }
  out << "]";
  return out;
}
std::ostream& operator<<(std::ostream& out, const Vector<3>& vec) {
  out << "[ ";
  for (int i = 0; i < 3; ++i) {
    out << vec[i] << "; ";
  }
  out << "]";
  return out;
}
std::istream& operator>>(std::istream& in, Vector<2>& vec) {
  for (int i = 0; i < 2; ++i) {
    in >> vec[i];
  }
  return in;
}
std::istream& operator>>(std::istream& in, Vector<3>& vec) {
  for (int i = 0; i < 3; ++i) {
    in >> vec[i];
  }
  return in;
}
/************************* Matrix Definition *************************/

template <int ROW, int COL>
Matrix<ROW, COL>::Matrix() {
  for (int i = 0; i < ROW; ++i) {
    mat_[i].fill(0);
  }
}

template <>
Matrix<2, 2>::Matrix(const std::array<double, 2>& vec0, const std::array<double, 2>& vec1) {
  mat_ = {vec0, vec1};
}

template <>
Matrix<3, 3>::Matrix(const std::array<double, 2>& vec0, const std::array<double, 2>& vec1) {
  OutLog("矩阵类错误使用，由二维数组无法构造三维矩阵");
  exit(1);
}

template <>
Matrix<2, 2>::Matrix(const Vector<2>& vec0, const Vector<2>& vec1) {
  for (int d = 0; d < 2; d++) {
    mat_[d][0] = vec0[d];
    mat_[d][1] = vec1[d];
  }
}

template <>
Matrix<3, 3>::Matrix(const Vector<2>& vec0, const Vector<2>& vec1) {
  OutLog("矩阵类错误使用，由二维数组无法构造三维矩阵");
  exit(1);
}

// template <>
// Matrix<2, 2>::Matrix(const Vector<2>& vec0, const Vector<2>& vec1,
//                     const Vector<2>& vec2) {
//  for (int d = 0; d < 2; d++) {
//    mat_[d][0] = vec0[d];
//    mat_[d][1] = vec1[d];
//  }
//}

template <>
Matrix<2, 2>::Matrix(const std::array<double, 3>& vec0, const std::array<double, 3>& vec1,
                     const std::array<double, 3>& vec2) {
  mat_[0][0] = 0;
  mat_[0][1] = 0;
  mat_[1][0] = 0;
  mat_[1][1] = 0;
}

template <>
Matrix<3, 3>::Matrix(const std::array<double, 3>& vec0, const std::array<double, 3>& vec1,
                     const std::array<double, 3>& vec2) {
  mat_ = {vec0, vec1, vec2};
}

template <>
Matrix<2, 2>::Matrix(const Vector<3>& vec0, const Vector<3>& vec1, const Vector<3>& vec2) {
  for (int d = 0; d < 2; d++) {
    mat_[d][0] = vec0[d];
    mat_[d][1] = vec1[d];
  }
}

template <>
Matrix<3, 3>::Matrix(const Vector<3>& vec0, const Vector<3>& vec1, const Vector<3>& vec2) {
  for (int d = 0; d < 3; d++) {
    mat_[d][0] = vec0[d];
    mat_[d][1] = vec1[d];
    mat_[d][2] = vec2[d];
  }
}

// template <>
// Matrix<3, 3>::Matrix(const Vector<2>& vec0, const Vector<2>& vec1,
//                     const Vector<2>& vec2) {
//  for (int d = 0; d < 2; d++) {
//    mat_[d][0] = vec0[d];
//    mat_[d][1] = vec1[d];
//    mat_[d][2] = vec2[d];
//  }
//  mat_[2][0] = 0;
//  mat_[2][1] = 0;
//  mat_[2][2] = 0;
//}

template <int ROW, int COL>
std::array<double, COL>& Matrix<ROW, COL>::operator[](int i) {
  return mat_[i];
}

template <int ROW, int COL>
const std::array<double, COL>& Matrix<ROW, COL>::operator[](int i) const {
  return mat_[i];
}

template <int ROW, int COL>
bool Matrix<ROW, COL>::operator==(const Matrix& mat) const {
  for (int i = 0; i < ROW; ++i) {
    if (mat_[i] != mat[i]) {
      return false;
    }
  }
  return true;
}

template <int ROW, int COL>
Matrix<ROW, COL> Matrix<ROW, COL>::operator-() const {
  Matrix<ROW, ROW> res;
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      res[i][j] = -mat_[i][j];
    }
  }
  return res;
}

template <int ROW, int COL>
Matrix<ROW, COL> Matrix<ROW, COL>::operator+(const Matrix& mat) const {
  Matrix<ROW, COL> res;
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      res[i][j] = mat_[i][j] + mat[i][j];
    }
  }
  return res;
}

template <int ROW, int COL>
Matrix<ROW, COL> Matrix<ROW, COL>::operator-(const Matrix& mat) const {
  Matrix<ROW, COL> res;
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      res[i][j] = mat_[i][j] - mat[i][j];
    }
  }
  return res;
}

template <int ROW, int COL>
Matrix<ROW, COL>& Matrix<ROW, COL>::operator+=(const Matrix& mat) {
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      mat_[i][j] += mat[i][j];
    }
  }
  return *this;
}

template <int ROW, int COL>
Matrix<ROW, COL>& Matrix<ROW, COL>::operator-=(const Matrix& mat) {
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      mat_[i][j] -= mat[i][j];
    }
  }
  return *this;
}

template <int ROW, int COL>
Matrix<ROW, COL> Matrix<ROW, COL>::operator*(double val) const {
  Matrix<ROW, COL> res;
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      res[i][j] = mat_[i][j] * val;
    }
  }
  return res;
}

template <int ROW, int COL>
Matrix<ROW, COL> Matrix<ROW, COL>::operator/(double val) const {
  Matrix<ROW, COL> res;
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      res[i][j] = mat_[i][j] / val;
    }
  }
  return res;
}

template <int ROW, int COL>
Matrix<ROW, COL>& Matrix<ROW, COL>::operator*=(double val) {
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      mat_[i][j] *= val;
    }
  }
  return *this;
}

template <int ROW, int COL>
Matrix<ROW, COL>& Matrix<ROW, COL>::operator/=(double val) {
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      mat_[i][j] /= val;
    }
  }
  return *this;
}

template <int ROW, int COL>
Vector<ROW> Matrix<ROW, COL>::operator*(const Vector<COL>& vec) const {
  Vector<ROW> res;
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      res[i] += mat_[i][j] * vec[j];
    }
  }
  return res;
}

template <int ROW, int COL>
Matrix<ROW, ROW> Matrix<ROW, COL>::operator*(const Matrix<COL, ROW>& mat) const {
  Matrix<ROW, ROW> res;
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < ROW; ++j) {
      for (int k = 0; k < COL; ++k) {
        res[i][j] += (mat_[i][k] * mat[k][j]);
      }
    }
  }
  return res;
}

template <int ROW, int COL>
Matrix<ROW, ROW>& Matrix<ROW, COL>::operator*=(const Matrix<COL, ROW>& mat) {
  Matrix<ROW, ROW> lmat = *this;
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < ROW; ++j) {
      mat_[i][j] = 0;
      for (int k = 0; k < COL; ++k) {
        mat_[i][j] += (lmat[i][k] * mat[k][j]);
      }
    }
  }
  return *this;
}

template <>
double Matrix<2, 2>::Determinant() const {
  return mat_[0][0] * mat_[1][1] - mat_[0][1] * mat_[1][0];
}

template <>
double Matrix<3, 3>::Determinant() const {
  return mat_[0][0] * mat_[1][1] * mat_[2][2] + mat_[0][2] * mat_[1][0] * mat_[2][1] +
         mat_[0][1] * mat_[1][2] * mat_[2][0] - mat_[0][2] * mat_[1][1] * mat_[2][0] -
         mat_[0][0] * mat_[1][2] * mat_[2][1] - mat_[0][1] * mat_[1][0] * mat_[2][2];
}

template <int ROW, int COL>
Matrix<COL, ROW> Matrix<ROW, COL>::Transpose() const {
  Matrix<COL, ROW> res;
  for (int i = 0; i < COL; ++i) {
    for (int j = 0; j < ROW; ++j) {
      res[i][j] = mat_[j][i];
    }
  }
  return res;
}

template <>
Matrix<2, 2> Matrix<2, 2>::Inverse() const {
  Matrix<2, 2> res;
  res[0][0] = mat_[1][1];
  res[0][1] = -mat_[0][1];

  res[1][0] = -mat_[1][0];
  res[1][1] = mat_[0][0];

  return res / this->Determinant();
}

template <>
Matrix<3, 3> Matrix<3, 3>::Inverse() const {
  Matrix<3, 3> res;
  res[0][0] = -mat_[1][2] * mat_[2][1] + mat_[1][1] * mat_[2][2];
  res[0][1] = mat_[0][2] * mat_[2][1] - mat_[0][1] * mat_[2][2];
  res[0][2] = -mat_[0][2] * mat_[1][1] + mat_[0][1] * mat_[1][2];

  res[1][0] = mat_[1][2] * mat_[2][0] - mat_[1][0] * mat_[2][2];
  res[1][1] = -mat_[0][2] * mat_[2][0] + mat_[0][0] * mat_[2][2];
  res[1][2] = mat_[0][2] * mat_[1][0] - mat_[0][0] * mat_[1][2];

  res[2][0] = -mat_[1][1] * mat_[2][0] + mat_[1][0] * mat_[2][1];
  res[2][1] = mat_[0][1] * mat_[2][0] - mat_[0][0] * mat_[2][1];
  res[2][2] = -mat_[0][1] * mat_[1][0] + mat_[0][0] * mat_[1][1];

  return res / this->Determinant();
}

template <int ROW, int COL>
Matrix<ROW, COL> Matrix<ROW, COL>::ZERO() {
  Matrix<ROW, COL> zero_mat;
  return zero_mat;
}

template <int ROW, int COL>
Matrix<ROW, COL> Matrix<ROW, COL>::UNIT() {
  Matrix<ROW, COL> unit_mat;
  for (int i = 0; i < ROW; ++i) {
    unit_mat[i][i] = 1;
  }
  return unit_mat;
}

template <int ROW, int COL>
Matrix<ROW, COL> Matrix<ROW, COL>::RAND(double low, double up) {
  // srand(time(nullptr) + std::rand());
  Matrix<ROW, COL> rand_mat;
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      rand_mat[i][j] = low + (up - low) * double(std::rand()) / (double(RAND_MAX) + 1.0);
    }
  }
  return rand_mat;
}

Matrix<2, 2> operator*(const double& val, const Matrix<2, 2>& mat) { return mat * val; }
Matrix<3, 3> operator*(const double& val, const Matrix<3, 3>& mat) { return mat * val; }
std::ostream& operator<<(std::ostream& out, const Matrix<2, 2>& mat) {
  out << "[";
  for (int i = 0; i < 2; ++i) {
    out << "[ ";
    for (int j = 0; j < 2; ++j) {
      out << mat[i][j] << ", ";
    }
    out << "]; ";
  }
  out << "]";
  return out;
}
std::ostream& operator<<(std::ostream& out, const Matrix<3, 3>& mat) {
  out << "[";
  for (int i = 0; i < 3; ++i) {
    out << "[ ";
    for (int j = 0; j < 3; ++j) {
      out << mat[i][j] << ", ";
    }
    out << "]; ";
  }
  out << "]";
  return out;
}
std::istream& operator>>(std::istream& in, Matrix<2, 2>& mat) {
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      in >> mat[i][j];
    }
  }
  return in;
}
std::istream& operator>>(std::istream& in, Matrix<3, 3>& mat) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      in >> mat[i][j];
    }
  }
  return in;
}
/************************* Angle Definition *************************/

Angle::Angle() { ang_ = 0.; }

Angle::Angle(double ang) { ang_ = ang; }

Angle::Angle(Vector<3> moment) { ang_ = moment[2]; }

double& Angle::operator()() { return ang_; }

const double& Angle::operator()() const { return ang_; }

Angle Angle::operator+(const Angle& ang) const {
  Angle res;
  res() = ang_ + ang();
  while (res() >= 2.0 * PI) res() -= 2.0 * PI;
  while (res() < 0) res() += 2.0 * PI;
  return res;
}

Angle& Angle::operator+=(const Angle& ang) {
  ang_ += ang();
  while (ang_ >= 2.0 * PI) ang_ -= 2.0 * PI;
  while (ang_ < 0) ang_ += 2.0 * PI;
  return *this;
}

Angle Angle::operator-(const Angle& ang) const {
  Angle res;
  res() = ang_ - ang();
  while (res() >= 2.0 * PI) res() -= 2.0 * PI;
  while (res() < 0) res() += 2.0 * PI;
  return res;
}

Angle& Angle::operator-=(const Angle& ang) {
  ang_ -= ang();
  while (ang_ >= 2.0 * PI) ang_ -= 2.0 * PI;
  while (ang_ < 0) ang_ += 2.0 * PI;
  return *this;
}

double Angle::RotAng() const { return ang_; }

Matrix<2, 2> Angle::RotMat() const {
  Matrix<2, 2> res;
  res[0][0] = std::cos(ang_);
  res[0][1] = -std::sin(ang_);

  res[1][0] = -res[0][1];
  res[1][1] = res[0][0];
  return res;
}

void Angle::Vec(Vector<2>& vec) {
  ang_ = atan2(vec[1], vec[0]);
  // std::cout << "ang_=" << ang_*180.0/PI << std::endl;
  while (ang_ >= 2.0 * PI) ang_ -= 2.0 * PI;
  while (ang_ < 0) ang_ += 2.0 * PI;
  // std::cout << "ang_=" << ang_ * 180.0 / PI << std::endl;
}

Angle Angle::ZERO() {
  Angle zero_ang;
  return zero_ang;
}

Angle Angle::Inverse() const {
  Angle ang;
  ang() = -this->ang_;
  return ang;
}

Angle Angle::RAND() {
  // srand(static_cast<int>(time(nullptr) + std::rand()));
  Angle rand_ang;
  rand_ang() = 2.0 * double(std::rand()) / (double(RAND_MAX) + 1) * PI;
  return rand_ang;
}

Angle Angle::RAND(double scale) {
  // srand(static_cast<int>(time(nullptr) + std::rand()));
  Angle rand_ang;
  rand_ang() = scale * 2.0 * double(std::rand()) / (double(RAND_MAX) + 1) * PI;
  return rand_ang;
}

Angle operator+(const double& val, const Angle& ang) { return ang + val; }

std::ostream& operator<<(std::ostream& out, const Angle& ang) {
  out << ang();
  return out;
}

std::istream& operator>>(std::istream& in, Angle& ang) {
  in >> ang();
  return in;
}

/************************* Quaternion Definition *************************/

Quaternion::Quaternion() {
  x_ = y_ = z_ = 0;
  w_ = 1;
}

Quaternion::Quaternion(double x, double y, double z, double w) {
  x_ = x;
  y_ = y;
  z_ = z;
  w_ = w;
}

/// @see
/// https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/
Quaternion::Quaternion(const Angle& ang, const Vector<3>& axi) {
  Vector<3> normalized = axi.Normalize();
  x_ = normalized[0] * sin(ang() / 2);
  y_ = normalized[1] * sin(ang() / 2);
  z_ = normalized[2] * sin(ang() / 2);
  w_ = cos(ang() / 2);
}

/// @see
/// https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
Quaternion::Quaternion(const Matrix<3, 3>& mat) {
  double trace = mat[0][0] + mat[1][1] + mat[2][2];
  if (trace > 0) {
    double s = 0.5 / std::sqrt(trace + 1.);
    w_ = 0.25 / s;
    x_ = (mat[2][1] - mat[1][2]) * s;
    y_ = (mat[0][2] - mat[2][0]) * s;
    z_ = (mat[1][0] - mat[0][1]) * s;
  } else {
    if (mat[0][0] > mat[1][1] && mat[0][0] > mat[2][2]) {
      double s = 2. * std::sqrt(1. + mat[0][0] - mat[1][1] - mat[2][2]);
      w_ = (mat[2][1] - mat[1][2]) / s;
      x_ = 0.25 * s;
      y_ = (mat[0][1] + mat[1][0]) / s;
      z_ = (mat[0][2] + mat[2][0]) / s;
    } else if (mat[1][1] > mat[2][2]) {
      double s = 2. * std::sqrt(1. + mat[1][1] - mat[0][0] - mat[2][2]);
      w_ = (mat[0][2] - mat[2][0]) / s;
      x_ = (mat[0][1] + mat[1][0]) / s;
      y_ = 0.25 * s;
      z_ = (mat[1][2] + mat[2][1]) / s;
    } else {
      double s = 2. * std::sqrt(1. + mat[2][2] - mat[0][0] - mat[1][1]);
      w_ = (mat[1][0] - mat[0][1]) / s;
      x_ = (mat[0][2] + mat[2][0]) / s;
      y_ = (mat[1][2] + mat[2][1]) / s;
      z_ = 0.25 * s;
    }
  }
}

Quaternion::Quaternion(const Vector<3>& moment) {
  double ang = moment.Norm();
  if (ang < 0.0000000001) {
    //OutLog("rotation moment is zero\n");
    x_ = 0;
    y_ = 0;
    z_ = 0;
    w_ = 1;
  } else {
    Vector<3> axi = moment.Normalize();
    x_ = axi[0] * sin(ang / 2);
    y_ = axi[1] * sin(ang / 2);
    z_ = axi[2] * sin(ang / 2);
    w_ = cos(ang / 2);
    this->Normalize();
  }
}

double& Quaternion::x() { return x_; }

double& Quaternion::y() { return y_; }

double& Quaternion::z() { return z_; }

double& Quaternion::w() { return w_; }

const double& Quaternion::x() const { return x_; }

const double& Quaternion::y() const { return y_; }

const double& Quaternion::z() const { return z_; }

const double& Quaternion::w() const { return w_; }

Vector<3> Quaternion::operator*(const Vector<3>& vec) const {
  Quaternion qua(vec[0], vec[1], vec[2], 0);
  Quaternion qua_res = (*this) * qua * (this->Inverse());
  return Vector<3>(qua_res.x(), qua_res.y(), qua_res.z());
}

/// @see https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
Quaternion Quaternion::operator*(const Quaternion& qua) const {
  Quaternion res;
  res.x() = w_ * qua.x() + x_ * qua.w() + y_ * qua.z() - z_ * qua.y();
  res.y() = w_ * qua.y() + y_ * qua.w() + z_ * qua.x() - x_ * qua.z();
  res.z() = w_ * qua.z() + z_ * qua.w() + x_ * qua.y() - y_ * qua.x();
  res.w() = w_ * qua.w() - x_ * qua.x() - y_ * qua.y() - z_ * qua.z();
  return res;
}

Quaternion& Quaternion::operator*=(const Quaternion& qua) {
  Quaternion lqua = *this;
  x_ = lqua.w() * qua.x() + lqua.x() * qua.w() + lqua.y() * qua.z() - lqua.z() * qua.y();
  y_ = lqua.w() * qua.y() + lqua.y() * qua.w() + lqua.z() * qua.x() - lqua.x() * qua.z();
  z_ = lqua.w() * qua.z() + lqua.z() * qua.w() + lqua.x() * qua.y() - lqua.y() * qua.x();
  w_ = lqua.w() * qua.w() - lqua.x() * qua.x() - lqua.y() * qua.y() - lqua.z() * qua.z();
  return *this;
}

double Quaternion::Norm() const { return std::sqrt(x_ * x_ + y_ * y_ + z_ * z_ + w_ * w_); }

/// @see
/// https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/
double Quaternion::RotAng() const { return 2. * std::acos(w_); }

/// @see
/// https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
Matrix<3, 3> Quaternion::RotMat() const {
  Matrix<3, 3> res;
  res[0][0] = 1 - 2 * y_ * y_ - 2 * z_ * z_;
  res[0][1] = 2 * x_ * y_ - 2 * z_ * w_;
  res[0][2] = 2 * x_ * z_ + 2 * y_ * w_;

  res[1][0] = 2 * x_ * y_ + 2 * z_ * w_;
  res[1][1] = 1 - 2 * x_ * x_ - 2 * z_ * z_;
  res[1][2] = 2 * y_ * z_ - 2 * x_ * w_;

  res[2][0] = 2 * x_ * z_ - 2 * y_ * w_;
  res[2][1] = 2 * y_ * z_ + 2 * x_ * w_;
  res[2][2] = 1 - 2 * x_ * x_ - 2 * y_ * y_;
  return res;
}

Quaternion Quaternion::Normalize() const {
  double norm = 1.0 / this->Norm();
  if (norm == 0.) {
    return *this;
  } else {
    return Quaternion(x_ * norm, y_ * norm, z_ * norm, w_ * norm);
  }
}

Quaternion Quaternion::Inverse() const {
  Quaternion normalized = this->Normalize();
  return Quaternion(-normalized.x(), -normalized.y(), -normalized.z(), normalized.w());
}

Quaternion Quaternion::ZERO() {
  Quaternion zero_qua;
  return zero_qua;
}

Quaternion Quaternion::RAND() {
  // srand(static_cast<int>(time(nullptr) + std::rand()));
  Quaternion rand_qua;
  rand_qua.x() = 2.0 * double(std::rand()) / (double(RAND_MAX) + 1.0) - 1.0;
  rand_qua.y() = 2.0 * double(std::rand()) / (double(RAND_MAX) + 1.0) - 1.0;
  rand_qua.z() = 2.0 * double(std::rand()) / (double(RAND_MAX) + 1.0) - 1.0;
  rand_qua.w() = 2.0 * double(std::rand()) / (double(RAND_MAX) + 1.0) - 1.0;
  return rand_qua.Normalize();
}

Quaternion Quaternion::RAND(double scale) {
  Quaternion rand_qua;
  // srand(static_cast<int>(time(nullptr) + std::rand()));
  Vector<3> normalized;
  double norm;
  do {
    normalized = Vector<3>::RAND(-1, 1);
    norm = normalized.Norm();
  } while (norm < 0.000001 || norm > 1);
  normalized.Normalize();
  double ang = scale * (PI * rand() / (double(RAND_MAX) + 1.0) - PI / 2.0);
  rand_qua.x() = normalized[0] * sin(ang);
  rand_qua.y() = normalized[1] * sin(ang);
  rand_qua.z() = normalized[2] * sin(ang);
  rand_qua.w() = cos(ang);
  return rand_qua.Normalize();
}

std::ostream& operator<<(std::ostream& out, const Quaternion& qua) {
  out << "[ " << qua.x() << "; " << qua.y() << "; " << qua.z() << "; " << qua.w() << "; ]";
  return out;
}

std::istream& operator>>(std::istream& in, Quaternion& qua) {
  in >> qua.x() >> qua.y() >> qua.z() >> qua.w();
  return in;
}

void PointTriangleDistance3D(const Vector<3>& point, const Vector<3>& triangle1,
                             const Vector<3>& triangle2, const Vector<3>& triangle3,
                             Vector<3>& result) {
  Vector<3> P, B, dd, e0, e1;
  double a, b, c, d, e, f, s, t;
  P = point;
  B = triangle1;
  e0 = triangle2 - triangle1;
  e1 = triangle3 - triangle1;
  dd = B - P;
  a = e0.Dot(e0);
  b = e0.Dot(e1);
  c = e1.Dot(e1);
  d = e0.Dot(dd);
  e = e1.Dot(dd);
  f = dd.Dot(dd);
  double det = abs(a * c - b * b);
  s = b * e - c * d;
  t = b * d - a * e;
  /*std::cout << "a=" << a << std::endl;
  std::cout << "b=" << b << std::endl;
  std::cout << "c=" << c << std::endl;
  std::cout << "d=" << d << std::endl;
  std::cout << "e=" << e << std::endl;
  std::cout << "s=" << s << std::endl;
  std::cout << "t=" << t << std::endl;
  std::cout << "s+t=" << s + t << std::endl;*/
  if (s + t <= det) {
    if (s < 0) {
      if (t < 0) {
        //region4;
        if (d < 0) {
          t = 0;
          s = (-d >= a ? 1 : -d / a);
        } else {
          s = 0;
          t = (e >= 0 ? 0 : (-e >= c ? 1 : -e / c));
        }
      } else {
        //regen3;
        s = 0;
        t = (e >= 0 ? 0 : (-e >= c ? 1 : -e / c));
      }
    } else {
      if (t < 0) {
        //regen 5;
        t = 0;
        s = (d >= 0 ? 0 : (-d >= a ? 1 : -d / a));
      } else {
        //regen0
        double invdet = 1.0 / det;
        s *= invdet;
        t *= invdet;
      }
    }
  } else {
    if (s < 0) {
      //regen2;
      double temp0 = b + d;
      double temp1 = c + e;
      if (temp1 > temp0) {
        double numer = temp1 - temp0;
        double denom = a - 2 * b + c;
        s = (numer >= denom ? 1 : numer / denom);
      } else {
        s = 0;
        t = (temp1 <= 0 ? 1 : (e >= 0 ? 0 : -e / c));
      }
    } else if (t < 0) {
      //regen6;
      double temp0 = b + e;
      double temp1 = a + d;
      if (temp1 > temp0) {
        double numer = temp1 - temp0;
        double denom = a - 2 * b + c;
        s = (numer >= denom ? 1 : numer / denom);
      } else {
        t = 0;
        s = (temp1 <= 0 ? 1 : (-d >= 0 ? 0 : -d / a));
      }
    } else {
      //regen1;
      double numer = c + e - b - d;
      if (numer <= 0) {
        s = 0;
      } else {
        double denom = a - 2.0 * b + c;
        s = (numer >= denom ? 1 : numer / denom);
      }
      t = 1 - s;
    }
  }
  result = B + s * e0 + t * e1;
}

void SegmentSegmentDistance3D(const Vector<3>& Spoint1, const Vector<3>& Spoint2,
                              const Vector<3>& Tpoint1, const Vector<3>& Tpoint2, Vector<3>& point1,
                              Vector<3>& point2) {
  Vector<3> d1 = Spoint2 - Spoint1;
  Vector<3> d2 = Tpoint2 - Tpoint1;
  //std::cout << "d1=" << d1 << std::endl;
  // std::cout << "d2=" << d2 << std::endl;
  auto u = Spoint1 - Tpoint1;
  double a = d1.Dot(d1);
  double b = d1.Dot(d2);
  double c = d2.Dot(d2);
  double d = d1.Dot(u);
  double e = d2.Dot(u);
  double det = a * c - b * b;

  //std::cout << "det=" << det << std::endl;
  double sNum = 0, tNum = 0, sDenom = 0, tDenom = 0;
  ///< 平行情况检查
  if (det < 0.0000000001) {
    //求点到直线的距离即可
    //std::cout << "平行颗粒\n";
    det = 0.0000000001;
    sNum = 0;
    tNum = e;
    tDenom = c;
    sDenom = det;
  } else {
    sNum = b * e - c * d;
    tNum = a * e - b * d;
  }
  //check s
  //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  //std::cout << "S=" << sNum << std::endl;
  //std::cout << "sdet=" << sDenom << std::endl;
  //std::cout << "t=" << tNum << std::endl;
  //std::cout << "td=" << tDenom << std::endl;
  sDenom = det;
  if (sNum < 0) {
    sNum = 0;
    tNum = e;
    tDenom = c;
  } else if (sNum > det) {
    sNum = det;
    tNum = e + b;
    tDenom = c;
  } else {
    tDenom = det;
  }
  //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  //std::cout << "S=" << sNum << std::endl;
  //std::cout << "sdet=" << sDenom << std::endl;
  //std::cout << "t=" << tNum << std::endl;
  //std::cout << "td=" << tDenom << std::endl;
  // check t
  if (tNum < 0) {
    tNum = 0;
    if (-d < 0) {
      sNum = 0;

    } else if (-d > a) {
      sNum = sDenom;
    } else {
      sNum = -d;
      sDenom = a;
    }
  } else if (tNum > tDenom) {
    tNum = tDenom;
    if ((-d + b) < 0) {
      sNum = 0;
    } else if ((-d + b) > a) {
      sNum = sDenom;
    } else {
      sNum = -d + b;
      sDenom = a;
    }
  }
  //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  //std::cout << "S=" << sNum << std::endl;
  //std::cout << "sdet=" << sDenom << std::endl;
  //std::cout << "t=" << tNum << std::endl;
  //std::cout << "td=" << tDenom << std::endl;
  double s = sNum / sDenom;
  double t = tNum / tDenom;
  //if (s > 1 || s < 0 || t > 1 || t < 0)
  //std::cout << "s=" << s << ",t=" << t << std::endl;

  point1 = Spoint1 + s * d1;
  point2 = Tpoint1 + t * d2;
}

double SegmentSegmentDistance2D(const Vector<2>& Spoint1, const Vector<2>& Spoint2,
                                const Vector<2>& Tpoint1, const Vector<2>& Tpoint2) {
  double b0, b1, a00, a11, a01;
  Vector<2> d0, d1;
  double t0, t1, t0_, t1_, t0__, t1__;  //t0,t1为交点对应参数，t0_,t1_,t__为特殊定义
  Vector<2> delta;
  delta = Spoint1 - Tpoint1;  //delta=p0-p1
  d0 = Spoint2 - Spoint1;
  d1 = Tpoint2 - Tpoint1;  //对应的T0,T1均为1

  Vector<2> d0n = d0.Normal(), d1n = d1.Normal();
  a00 = d0.Dot(d0);
  a01 = d0.Dot(d1);
  a11 = d1.Dot(d1);
  b0 = d0.Dot(delta);
  b1 = d1.Dot(delta);
  t0_ = -b0 / a00;
  t1_ = b1 / a11;
  t0__ = (a01 - b0) / a00;
  t1__ = (a01 + b1) / a11;
  //平行检测
  double det = (d0 / d0.Norm()).Dot(d0 / d0.Norm()) * (d1 / d1.Norm()).Dot((d1 / d1.Norm())) -
               (d0 / d0.Norm()).Dot((d1 / d1.Norm())) * (d0 / d0.Norm()).Dot((d1 / d1.Norm()));
  if (fabs(det) < 1e-7) {
    if (a01 < 0 && d0.Dot(delta) >= 0) {
      return delta.Norm();
    } else if (a01 > 0 && d0.Dot(delta + d0) <= 0) {
      return (delta + d0).Norm();
    } else if (a01 > 0 && d0.Dot(delta - d1) >= 0) {
      return (delta - d1).Norm();
    } else if (a01 < 0 && d0.Dot(delta + d0 - d1) <= 0) {
      return (delta + d0 - d1).Norm();
    } else {
      return fabs(d0n.Dot(delta)) / d0.Norm();
    }
  }  //两条线段平行
  t0 = -d1n.Dot(delta) / d1n.Dot(d0);
  t1 = -d0n.Dot(delta) / d1n.Dot(d0);
  //若两条线段不平行
  //std::cout << t0 << ' ' << t1 << std::endl;
  //std::cout << t0_ << ' ' << t1_ << std::endl;
  //std::cout << t0__ << ' ' << t1__ << std::endl;
  //std::cout << fabs(d0n.Dot(delta - d1)) / d0.Norm() << std::endl;
  if (t0 > 0 && t0 < 1 && t1 > 0 && t1 < 1) {
    return 0;
  } else if (t0_ > 0 && t0_ < 1 && t1 <= 0)
    return fabs(d0n.Dot(delta)) / d0.Norm();
  else if (t0__ > 0 && t0__ < 1 && t1 >= 1)
    return fabs(d0n.Dot(delta - d1)) / d0.Norm();
  else if (t1_ > 0 && t1_ < 1 && t0 <= 0)
    return fabs(d1n.Dot(delta)) / d1.Norm();
  else if (t1__ > 0 && t1__ < 1 && t0 >= 1)
    return fabs(d1n.Dot(delta + d0)) / d1.Norm();
  else if (t0_ <= 0 && t1_ <= 0)
    return delta.Norm();
  else if (t0_ >= 1 && t1_ <= 0)
    return (delta + d0).Norm();
  else if (t0_ <= 0 && t1_ >= 1)
    return (delta - d1).Norm();
  else if (t0_ >= 1 && t1_ >= 1)
    return (delta + d0 - d1).Norm();
}
double SegmentSegmentAngle2D(const Vector<2>& Spoint1, const Vector<2>& Spoint2,
                             const Vector<2>& Tpoint1, const Vector<2>& Tpoint2) {
  Vector<2> a = Spoint2 - Spoint1, b = Tpoint2 - Tpoint1;
  //std::cout << a.Dot(b) <<' '<<a.Norm()<<' '<<b.Norm()<< std::endl;
  double cos = fabs(a.Dot(b)) / (a.Norm() * b.Norm());
  double theta = acos(cos);
  return theta;
}

void PointSegmentDistance3D(const Vector<3>& point, const Vector<3>& segment0,
                            const Vector<3>& segment1, Vector<3>& result) {
  Vector<3> D = segment1 - segment0;
  Vector<3> YmP0 = point - segment0;
  double t = D.Dot(YmP0);
  if (t <= 0) {
    result = segment0;
    return;
  }
  double DdD = D.Dot(D);
  if (t >= DdD) {
    result = segment1;
    return;
  }
  result = segment0 + t * D / DdD;
  return;
}

double random_double(double m, double n) {
  double result = n + double(std::rand()) / (double(RAND_MAX) + 1) * (m - n);
  return result;
}

int random_int(int m, int n) {
  std::uniform_int_distribution<int> u(m, n);
  std::default_random_engine e;
  e.seed(time(NULL));
  return u(e);
}
int Heaviside(double x){
  return (x > 0) ? 1 : 0;
}
int sign(double x) {
  if (x == 0)
    return 0;
  else
    return (x > 0) ? 1 : -1;
}
void OutLog(char* szLog) {
  std::time_t t = std::time(0);
  std::tm* st = std::localtime(&t);
  FILE* fp;
  fp = fopen("log.txt", "at");
  fprintf(fp, "%d%d%d:%d:%d:%d: %s\n ", st->tm_year, st->tm_mon, st->tm_mday, st->tm_hour,
          st->tm_min, st->tm_sec, szLog);
  fclose(fp);
};
void OutLog(std::string str) {
  std::time_t t = std::time(0);
  std::tm* st = std::localtime(&t);
  FILE* fp;
  fp = fopen("log.txt", "at");
  fprintf(fp, "%d%d%d:%d:%d:%d: %s\n", st->tm_year, st->tm_mon, st->tm_mday, st->tm_hour,
          st->tm_min, st->tm_sec, str.c_str());
  fclose(fp);
};
void OutLog(const char* szLog) {
  std::time_t t = std::time(0);
  std::tm* st = std::localtime(&t);
  FILE* fp;
  fp = fopen("log.txt", "at");
  fprintf(fp, "%d%d%d:%d:%d:%d: %s\n ", st->tm_year + 1900, st->tm_mon + 1, st->tm_mday,
          st->tm_hour, st->tm_min, st->tm_sec, szLog);
  fclose(fp);
};

template class Vector<2>;
template class Vector<3>;
template class Matrix<2, 2>;
template class Matrix<3, 3>;
