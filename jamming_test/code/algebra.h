/**
 * @file algebra.h
 * @author 李水乡课题组
 * @brief 代数库
 * @version 0.1
 * @date 2021-09-04
 *
 * @copyright Copyright (c) 2021
 *
 * @example example/algebra_example.cpp
 *
 */

#ifndef PACKING_ALGEBRA_H_
#define PACKING_ALGEBRA_H_

#include <array>
#include <cmath>
#include <ctime>
#include <iostream>
#include <random>
#include <time.h>
#include "sys/timeb.h"
#include <functional>
template <int ROW>
class Vector;
template <int ROW, int COL>
class Matrix;
class Angle;
class Quaternion;

constexpr double PI = 3.14159265358979324;  ///< 圆周率
constexpr double INF = 1E100;               ///< 无穷大

/************************* Vector Declaration *************************/
/**
 * @brief 向量类
 * @details
 * 实现向量和实数的乘除，向量的加减、点积、叉积、双击，向量的取模、单位化等。
 */
template <int ROW>
class Vector {
 private:
  std::array<double, ROW> vec_;

 public:
  Vector();                              ///< 默认构造函数生成零向量
  Vector(double x, double y);            ///< 构造二维向量
  Vector(double x, double y, double z);  ///< 构造三维向量
  Vector(std::array<double, ROW> vec);

  double& operator[](int i);                 ///< 向量取值/赋值
  const double& operator[](int i) const;     ///< 常向量取值
  bool operator==(const Vector& vec) const;  ///< 向量相等判断

  Vector operator-() const;                   ///< 负向量
  Vector operator+(const Vector& vec) const;  ///< 向量元素相加
  Vector operator-(const Vector& vec) const;  ///< 向量元素相减
  Vector& operator+=(const Vector& vec);      ///< 向量元素相加
  Vector& operator-=(const Vector& vec);      ///< 向量元素相减

  Vector operator*(double val) const;  ///< 向量实数相乘
  Vector operator/(double val) const;  ///< 向量实数相除
  Vector& operator*=(double val);      ///< 向量实数相乘
  Vector& operator/=(double val);      ///< 向量实数相除

  double Dot(const Vector& vec) const;               ///< 向量点积
  Vector<3> Cross(const Vector& vec) const;          ///< 向量叉积
  Matrix<ROW, ROW> Dyadic(const Vector& vec) const;  ///< 向量双积

  double MaxElement() const;  ///<取向量中的最大元素
  double Norm() const;        ///< 向量取模
  Vector<2> Normal() const;   ///<求二维的法向量
  Vector Normalize() const;   ///< 向量单位化

  static Vector ZERO();                                 ///< 零向量
  static Vector UNIT();                                 ///< 1向量
  static Vector RAND(double low = 0., double up = 1.);  ///< 随机向量
  static Vector RAND(const Matrix<ROW, ROW>& mat);      ///< 随机向量
};

Vector<2> operator*(const double& val, const Vector<2>& vec);
Vector<3> operator*(const double& val, const Vector<3>& vec);

std::ostream& operator<<(std::ostream& out, const Vector<2>& vec);
std::ostream& operator<<(std::ostream& out, const Vector<3>& vec);

std::istream& operator>>(std::istream& in, Vector<2>& vec);
std::istream& operator>>(std::istream& in, Vector<3>& vec);

template <>
Vector<2>::Vector(double x, double y, double z);

template <>
Vector<3>::Vector(double x, double y, double z);

/************************* Matrix Declaration *************************/

/**
 * @brief 矩阵类
 * @details
 * 实现矩阵和实数的乘除，矩阵和向量的乘法，矩阵的加减、乘法，矩阵行列式、转置、求逆等。
 */
template <int ROW, int COL>
class Matrix {
 private:
  std::array<std::array<double, COL>, ROW> mat_;

 public:
  Matrix();  ///< 默认构造函数生成零矩阵
  Matrix(const std::array<double, 2>& vec0, const std::array<double, 2>& vec1);  ///< 构造2×2矩阵
  Matrix(const Vector<2>& vec0, const Vector<2>& vec1);  ///< 构造2×2矩阵
  Matrix(const std::array<double, 3>& vec1, const std::array<double, 3>& vec2,
         const std::array<double, 3>& vec3);  ///< 构造3×3矩阵
  Matrix(const Vector<3>& vec0, const Vector<3>& vec1, const Vector<3>& vec2);  ///< 构造3×3矩阵

  std::array<double, COL>& operator[](int i);              ///< 矩阵取值/赋值
  const std::array<double, COL>& operator[](int i) const;  ///< 常矩阵取值
  bool operator==(const Matrix& mat) const;                ///< 矩阵相等判断

  Matrix operator-() const;                   ///< 负矩阵
  Matrix operator+(const Matrix& vec) const;  ///< 矩阵元素相加
  Matrix operator-(const Matrix& vec) const;  ///< 矩阵元素相减
  Matrix& operator+=(const Matrix& vec);      ///< 矩阵元素相加
  Matrix& operator-=(const Matrix& vec);      ///< 矩阵元素相减

  Matrix operator*(double val) const;  ///< 矩阵实数相乘
  Matrix operator/(double val) const;  ///< 矩阵实数相除
  Matrix& operator*=(double val);      ///< 矩阵实数相乘
  Matrix& operator/=(double val);      ///< 矩阵实数相除

  Vector<ROW> operator*(const Vector<COL>& vec) const;            ///< 矩阵向量乘积
  Matrix<ROW, ROW>& operator*=(const Matrix<COL, ROW>& mat);      ///< 矩阵乘积
  Matrix<ROW, ROW> operator*(const Matrix<COL, ROW>& mat) const;  ///< 矩阵乘积

  double Determinant() const;          ///< 矩阵行列式
  Matrix<COL, ROW> Transpose() const;  ///< 矩阵转置
  Matrix Inverse() const;              ///< 矩阵求逆

  static Matrix ZERO();                                 ///< 零矩阵
  static Matrix UNIT();                                 ///< 单位矩阵
  static Matrix RAND(double low = 0., double up = 1.);  ///< 随机矩阵
};

Matrix<2, 2> operator*(const double& val, const Matrix<2, 2>& mat);
Matrix<3, 3> operator*(const double& val, const Matrix<3, 3>& mat);
std::ostream& operator<<(std::ostream& out, const Matrix<2, 2>& mat);
std::ostream& operator<<(std::ostream& out, const Matrix<3, 3>& mat);
std::istream& operator>>(std::istream& in, Matrix<2, 2>& mat);
std::istream& operator>>(std::istream& in, Matrix<3, 3>& mat);

/************************* Angle Declaration *************************/

/**
 * @brief 角度类
 * @details 实现二维转角的加法，角度转矩阵等。
 *
 */
class Angle {
 private:
  double ang_;

 public:
  Angle();
  Angle(double ang);
  Angle(Vector<3> moment);

  double& operator()();              ///< 角度取值/赋值
  const double& operator()() const;  ///< 常角度取值

  Angle operator+(const Angle& ang) const;  ///< 角度相加
  Angle& operator+=(const Angle& ang);      ///< 角度相加
  Angle operator-(const Angle& ang) const;  ///< 角度相加
  Angle& operator-=(const Angle& ang);      ///< 角度相加

  Matrix<2, 2> RotMat() const;  ///< 角度转矩阵
  void Vec(Vector<2>& vec);     ///< 通过向量初始化角度
  double RotAng() const;        ///< 转角度

  static Angle ZERO();  ///< 零角度
  Angle Inverse() const;
  static Angle RAND();              ///< 随机角度
  static Angle RAND(double scale);  ///< 随机角度
};

Angle operator+(const double& val, const Angle& ang);  ///< 实数加转角
std::ostream& operator<<(std::ostream& out, const Angle& ang);
std::istream& operator>>(std::istream& in, Angle& ang);
/************************* Quaternion Declaration *************************/

/**
 * @brief 四元数类
 * @details 实现四元数旋转向量，四元数乘法，四元数取模、转矩阵、单位化、求逆等。
 *
 */
class Quaternion {
 private:
  double x_, y_, z_, w_;

 public:
  Quaternion();  ///< 默认构造函数生成零四元数
  Quaternion(double x, double y, double z,
             double w);  ///< 通过四个元素构造四元数
  Quaternion(const Angle& ang, const Vector<3>& axi = Vector<3>(0, 0,
                                                                1));  ///< 通过旋转角度构造四元数
  Quaternion(const Matrix<3, 3>& mat);  ///< 通过旋转矩阵构造四元数

  Quaternion(const Vector<3>& moment);  ///< 通过力矩构造旋转量

  double& x();  ///< 四元数取值/赋值
  double& y();  ///< 四元数取值/赋值
  double& z();  ///< 四元数取值/赋值
  double& w();  ///< 四元数取值/赋值

  const double& x() const;  ///< 常四元数取值
  const double& y() const;  ///< 常四元数取值
  const double& z() const;  ///< 常四元数取值
  const double& w() const;  ///< 常四元数取值

  Vector<3> operator*(const Vector<3>& vec) const;  ///< 四元数作用于向量

  Quaternion operator*(const Quaternion& qua) const;  ///< 四元数乘积
  Quaternion& operator*=(const Quaternion& qua);      ///< 四元数乘积

  double Norm() const;           ///< 四元数取模
  double RotAng() const;         ///< 四元数转角度
  Matrix<3, 3> RotMat() const;   ///< 四元数转矩阵
  Quaternion Normalize() const;  ///< 四元数单位化
  Quaternion Inverse() const;    ///< 四元数求逆

  static Quaternion ZERO();              ///< 零四元数
  static Quaternion RAND();              ///< 随机四元数
  static Quaternion RAND(double scale);  ///< 随机四元数
};

std::ostream& operator<<(std::ostream& out, const Quaternion& qua);

std::istream& operator>>(std::istream& in, Quaternion& qua);

///基本几何运算函数
/// 线段之间的夹角函数
double SegmentSegmentAngle2D(const Vector<2>& Spoint1, const Vector<2>& Spoint2,
                             const Vector<2>& Tpoint1, const Vector<2>& Tpoint2);
/// 线段之间的距离函数
void SegmentSegmentDistance3D(const Vector<3>& Spoint1, const Vector<3>& Spoint2,
                              const Vector<3>& Tpoint1, const Vector<3>& Tpoint2, Vector<3>& point1,
                              Vector<3>& point2);
double SegmentSegmentDistance2D(const Vector<2>& Spoint1, const Vector<2>& Spoint2,
                                const Vector<2>& Tpoint1, const Vector<2>& Tpoint2);

/// <summary>
/// Minimum distance from point to triangle
/// </summary>
/// <param name="point"></param> The coordinates of point
/// <param name="triangle1"></param> Vertex 1 of the triangle
/// <param name="triangle2"></param> Vertex 2 of the triangle
/// <param name="triangle3"></param> Vertex 3 of the triangle
/// <param name="result"></param> The closest point on the triangle
void PointTriangleDistance3D(const Vector<3>& point, const Vector<3>& triangle1,
                             const Vector<3>& triangle2, const Vector<3>& triangle3,
                             Vector<3>& result);

void PointSegmentDistance3D(const Vector<3>& point, const Vector<3>& segment1,
                            const Vector<3>& segment2, Vector<3>& result);
///生成范围内随机数
double random_double(double m, double n);
int random_int(int m, int n);

///符号函数
int sign(double x);
/// <summary>
/// 日志文件的输出系列函数
/// </summary>
/// <param name="szLog"></param>
void OutLog(char* szLog);
void OutLog(std::string str);
void OutLog(const char* szLog);

#endif  // PACKING_ALGEBRA_H_
