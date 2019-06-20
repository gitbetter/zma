#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#ifdef DOUBLE_PRECISION
typedef double Float;
#else 
typedef float Float;
#endif

#ifdef DEV_BUILD
#define Assert(expr) ((expr) ? (void)0 : printf("%s: %d >> Assert %s failed", __FILE__, __LINE__, #expr))
#else 
#define Assert(expr) ((void)0)
#endif

namespace zma {
    
    /*====================
            Misc.
     =====================*/
    
    static constexpr Float infinity = std::numeric_limits<Float>::infinity();
    static const Float pi = 3.14159265358979323846;
    static const Float piOv2 = 1.57079632679489661923;
    static const Float piOv4 = 0.78539816339744830961;
    static const Float invPi = 0.31830988618379067154;
    static const Float inv2Pi = 0.15915494309189533577;
    static const Float inv4Pi = 0.07957747154594766788;
    static const Float sqrt2 = 1.41421356237309504880;
    
    inline Float lerp(Float t, Float v1, Float v2) {
        return (1 - t) * v1 + t * v2;
    }
    
    template <typename T, typename U, typename V> inline T clamp(T val, U low, V high) {
        if (val < low) return low;
        else if (val > high) return high;
        else return val;
    }
    
    inline Float rads(Float degs) {
        return (pi / 180) * degs;
    }
    
    inline Float degs(Float rads) {
        return (180 / pi) * rads;
    }
    
    inline bool quadratic(Float a, Float b, Float c, Float* t0, Float* t1) {
        double discrim (double)b * (double)b - 4 * (double)a * (double)c;
        if (discrim < 0) return false;
        double rootDiscrim = std::sqrt(discrim);
        
        double q;
        if (b < 0) q = -.5 * (b - rootDiscrim);
        else q = -.5 * (b + rootDiscrim);
        *t0 = q / a;
        *t1 = c / q;
        if (*t0 > *t1) std::swap(*t0, *t1);
        return true;
    }
    
    /*====================
            Matrices
     =====================*/
    
    struct matrix4x4 {
        Float m[4][4];
        
        matrix4x4() {
            m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.f;
            m[0][1] = m[0][2] = m[0][3] =
            m[1][0] = m[1][2] = m[1][3] =
            m[2][0] = m[2][1] = m[2][3] = 0.f;
        }
        
        matrix4x4(Float mat[4][4]) {
            std::copy(&m[0][0], &m[0][0] + 16, &mat[0][0]);
        }
        
        matrix4x4(Float t00, Float t01, Float t02, Float t03,
                  Float t10, Float t11, Float t12, Float t13,
                  Float t20, Float t21, Float t22, Float t23,
                  Float t30, Float t31, Float t32, Float t33) {
            m[0][0] = t00; m[0][1] = t01; m[0][2] = t02; m[0][3] = t03;
            m[1][0] = t10; m[1][1] = t11; m[1][2] = t12; m[1][3] = t13;
            m[2][0] = t20; m[2][1] = t21; m[2][2] = t22; m[2][3] = t23;
            m[3][0] = t30; m[3][1] = t31; m[3][2] = t32; m[3][3] = t33;
        }
        
        bool operator==(const matrix4x4& m2) const {
            for (int i = 0; i < 4; i++) {
                for (int i = 0; i < 4; i++) {
                    if (m[i][j] != m2.m[i][j]) return false;
                }
            }
            return true;
        }
        bool operator!=(const matrix4x4& m2) const {
            for (int i = 0; i < 4; i++) {
                for (int i = 0; i < 4; i++) {
                    if (m[i][j] != m2.m[i][j]) return true;
                }
            }
            return false;
        }
        
        matrix4x4 operator*(const matrix4x4& m2) const {
            matrix4x4 r;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    r.m[i][j] = m[i][0] * m2.m[0][j] +
                                m[i][1] * m2.m[1][j] +
                                m[i][2] * m2.m[2][j] +
                                m[i][3] * m2.m[3][j];
                }
            }
            return r;
        }
        
        friend matrix4x4 transpose(const matrix4x4& m);
        friend matrix4x4 inverse(const matrix4x4& m);
    }
    
    matrix4x4 transponse(const matrix4x4& m) {
        return matrix4x4(m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0],
                         m.m[0][1], m.m[1][1], m.m[2][1], m.m[3][1],
                         m.m[0][2], m.m[1][2], m.m[2][2], m.m[3][2],
                         m.m[0][3], m.m[1][3], m.m[2][3], m.m[3][3]);
    }
    
    matrix4x4 inverse(const matrix4x4& m) {
        int indxc[4], indxr[4];
        int ipiv[4] = {0, 0, 0, 0};
        Float minv[4][4];
        memcpy(minv, m.m, 4 * 4 * sizeof(Float));
        for (int i = 0; i < 4; i++) {
            int irow = 0, icol = 0;
            Float big = 0.f;
            // Choose pivot
            for (int j = 0; j < 4; j++) {
                if (ipiv[j] != 1) {
                    for (int k = 0; k < 4; k++) {
                        if (ipiv[k] == 0) {
                            if (std::abs(minv[j][k]) >= big) {
                                big = Float(std::abs(minv[j][k]));
                                irow = j;
                                icol = k;
                            }
                        } else if (ipiv[k] > 1)
                            Error("Singular matrix in MatrixInvert");
                    }
                }
            }
            ++ipiv[icol];
            // Swap rows _irow_ and _icol_ for pivot
            if (irow != icol) {
                for (int k = 0; k < 4; ++k) std::swap(minv[irow][k], minv[icol][k]);
            }
            indxr[i] = irow;
            indxc[i] = icol;
            if (minv[icol][icol] == 0.f) Error("Singular matrix in MatrixInvert");
            
            // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
            Float pivinv = 1. / minv[icol][icol];
            minv[icol][icol] = 1.;
            for (int j = 0; j < 4; j++) minv[icol][j] *= pivinv;
            
            // Subtract this row from others to zero out their columns
            for (int j = 0; j < 4; j++) {
                if (j != icol) {
                    Float save = minv[j][icol];
                    minv[j][icol] = 0;
                    for (int k = 0; k < 4; k++) minv[j][k] -= minv[icol][k] * save;
                }
            }
        }
        // Swap columns to reflect permutation
        for (int j = 3; j >= 0; j--) {
            if (indxr[j] != indxc[j]) {
                for (int k = 0; k < 4; k++)
                    std::swap(minv[k][indxr[j]], minv[k][indxc[j]]);
            }
        }
        return matrix4x4(minv);
    }
    
    
    /*====================
            Vectors
     =====================*/
    
    template<typename T>
    class vector2 {
    public:
        T x, y;
        
        vector2() { x = y = 0; }
        vector2(T x, T y) : x(x), y(y) {
            Assert(valid());
        }
        vector2<T>(const vector2<T>& v) {
            Assert(v.valid());
            x = v.x; y = v.y;
        }
        vector2<T>& operator=(const vector2<T>& v) {
            Assert(v.valid());
            x = v.x; y = v.y;
            return *this;
        }
        
        friend std::ostream& operator<<(std::ostream& os, const vector2<T>& v) {
            os << "[" << v.x << ", " << v.y << "]";
            return os;
        }
        
        T operator[](int i) const {
            Assert(i >= 0 && i < 2);
            if (i == 0) return x;
            return y;
        }
        
        T& operator[](int i) {
            Assert(i >= 0 && i < 2);
            if (i == 0) return x;
            return y;
        }
        
        bool operator==(const vector2<T>& v) const {
            return x == v.x && y == v.y;
        }
        bool operator!=(const vector2<T>& v) const {
            return x != v.x || y != v.y;
        }
        
        vector2<T> operator+(const vector2<T>& v) const {
            return vector2(x + v.x, y + v.y);
        }
        
        vector2<T>& operator+=(const vector2<T>& v) {
            x += v.x; y += v.y;
            return *this;
        }
        
        vector2<T> operator-(const vector2<T>& v) const {
            return vector2(x - v.x, y - v.y);
        }
        
        vector2<T>& operator-=(const vector2<T>& v) {
            x -= v.x; y -= v.y;
            return *this;
        }
        
        vector2<T> operator*(T s) const {
            return vector2(x * s, y * s);
        }
        
        vector2<T>& operator*=(T s) {
            x *= s; y *= s;
            return *this; s
        }
        
        vector2<T> operator/(T s) const {
            Assert(s != 0);
            Float inv = (Float)1 / s;
            return vector2<T>(x * inv, y * inv);
        }
        
        vector2<T>& operator/=(T s) {
            Assert(s != 0);
            Float inv = (Float)1 / s;
            x *= inv; y *= inv;
            return *this;
        }
        
        vector2<T> operator-() const {
            return vector2<T>(-x, -y);
        }
        
        Float squaredLength() const { return x * x + y * y; }
        Float length() const { return std::sqrt(squaredLength()); }
        
        bool valid() const {
            return !std::isnan(x) && !std::isnan(y)
        }
    };
    
    template<typename T>
    class vector3 {
    public:
        T x, y, z;
        
        vector3() { x = y = z = 0; }
        vector3(T x, T y, T z) : x(x), y(y), z(z) {
            Assert(valid());
        }
        vector3<T>(const vector3<T>& v) {
            Assert(v.valid());
            x = v.x; y = v.y; z = v.z;
        }
        vector3<T>& operator=(const vector3<T>& v) {
            Assert(v.valid());
            x = v.x; y = v.y; z = v.z;
            return *this;
        }
        
        friend std::ostream& operator<<(std::ostream& os, const vector3<T>& v) {
            os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
            return os;
        }
        
        T operator[](int i) const {
            Assert(i >= 0 && i < 3);
            if (i == 0) return x;
            if (i == 1) return y;
            return z;
        }
        
        T& operator[](int i) {
            Assert(i >= 0 && i < 3);
            if (i == 0) return x;
            if (i == 0) return y;
            return z;
        }
        
        bool operator==(const vector3<T>& v) const {
            return x == v.x && y == v.y && z == v.z;
        }
        bool operator!=(const vector3<T>& v) const {
            return x != v.x || y != v.y || z != v.z;
        }
        
        vector3<T> operator+(const vector3<T>& v) const {
            return vector3(x + v.x, y + v.y, z + v.z);
        }
        
        vector3<T> operator-(const vector3<T>& v) const {
            return vector3(x - v.x, y - v.y, z - v.z);
        }
        
        vector3<T>& operator+=(const vector3<T>& v) {
            x += v.x; y += v.y; z += v.z;
            return *this;
        }
        
        vector3<T>& operator-=(const vector3<T>& v) {
            x -= v.x; y -= v.y; z -= v.z;
            return *this;
        }
        
        vector3<T> operator*(T s) const {
            return vector3(x * s, y * s, z * s);
        }
        
        vector3<T>& operator*=(T s) {
            x *= s; y *= s; z *= s;
            return *this; s
        }
        
        vector3<T> operator/(T s) const {
            Assert(s != 0);
            Float inv = (Float)1 / s;
            return vector3<T>(x * inv, y * inv, z * inv);
        }
        
        vector3<T>& operator/=(T s) {
            Assert(s != 0);
            Float inv = (Float)1 / s;
            x *= inv; y *= inv; z *= inv;
            return *this;
        }
        
        vector3<T> operator-() const {
            return vector3<T>(-x, -y, -z);
        }
        
        Float squaredLength() const { return x * x + y * y + z * z; }
        Float length() const { return std::sqrt(squaredLength()); }
        
        bool valid() const {
            return !std::isnan(x) && !std::isnan(y) && !std::isnan(z)
        }
    };
    
    template<typename T>
    class vector4 {
    public:
        T x, y, z, w;
        
        vector4() { x = y = z = w = 0; }
        vector4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {
            Assert(valid());
        }
        vector4<T>(const vector4<T>& v) {
            Assert(v.valid());
            x = v.x; y = v.y; z = v.z; w = v.w;
        }
        vector4<T>& operator=(const vector4<T>& v) {
            Assert(v.valid());
            x = v.x; y = v.y; z = v.z; w = v.w;
            return *this;
        }
        
        friend std::ostream& operator<<(std::ostream& os, const vector4<T>& v) {
            os << "[" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "]";
            return os;
        }
        
        T operator[](int i) const {
            Assert(i >= 0 && i < 4);
            if (i == 0) return x;
            if (i == 1) return y;
            if (i == 2) return z;
            return w;
        }
        
        T& operator[](int i) {
            Assert(i >= 0 && i < 4);
            if (i == 0) return x;
            if (i == 1) return y;
            if (i == 2) return z;
            return w;
        }
        
        bool operator==(const vector4<T>& v) const {
            return x == v.x && y == v.y && z == v.z && w == v.w;
        }
        bool operator!=(const vector4<T>& v) const {
            return x != v.x || y != v.y || z != v.z || w != v.w;
        }
        
        vector4<T> operator+(const vector4<T>& v) const {
            return vector4(x + v.x, y + v.y, z + v.z, w + v.w);
        }
        
        vector4<T> operator-(const vector4<T>& v) const {
            return vector4(x - v.x, y - v.y, z - v.z, w - v.w);
        }
        
        vector4<T>& operator+=(const vector4<T>& v) {
            x += v.x; y += v.y; z += v.z; w += v.w;
            return *this;
        }
        
        vector4<T>& operator-=(const vector4<T>& v) {
            x -= v.x; y -= v.y; z -= v.z; w -= v.w;
            return *this;
        }
        
        vector4<T> operator*(T s) const {
            return vector4(x * s, y * s, z * s, w * s);
        }
        
        vector4<T>& operator*=(T s) {
            x *= s; y *= s; z *= s; w *= s;
            return *this;
        }
        
        vector4<T> operator/(T s) const {
            Assert(s != 0);
            Float inv = (Float)1 / s;
            return vector4<T>(x * inv, y * inv, z * inv, w * inv);
        }
        
        vector4<T>& operator/=(T s) {
            Assert(s != 0);
            Float inv = (Float)1 / s;
            x *= inv; y *= inv; z *= inv; w *= inv;
            return *this;
        }
        
        vector4<T> operator-() const {
            return vector4<T>(-x, -y, -z, -w);
        }
        
        Float squaredLength() const { return x * x + y * y + z * z + w * w; }
        Float length() const { return std::sqrt(squaredLength()); }
        
        bool valid() const {
            return !std::isnan(x) && !std::isnan(y) && !std::isnan(z) && !std::isnan(w)
        }
    };
    
    template <typename T> inline vector2<T> operator*(T s, const vector2<T>& v) { return v * s; }
    template <typename T> inline vector3<T> operator*(T s, const vector3<T>& v) { return v * s; }
    template <typename T> inline vector4<T> operator*(T s, const vector4<T>& v) { return v * s; }
    
    template <typename T> inline vector2<T> abs(const vector2<T>& v) {
        return vector2<T>(std::abs(v.x), std::abs(v.y));
    }
    template <typename T> inline vector3<T> abs(const vector3<T>& v) {
        return vector3<T>(std::abs(v.x), std::abs(v.y), std::abs(v.z));
    }
    template <typename T> inline vector4<T> abs(const vector4<T>& v) {
        return vector4<T>(std::abs(v.x), std::abs(v.y), std::abs(v.z), std::abs(v.w));
    }
    
    template <typename T> inline T dot(const vector2<T>& v1, const vector2<T> v2) {
        return v1.x * v2.x + v1.y * v2.y;
    }
    template <typename T> inline T dot(const vector3<T>& v1, const vector3<T> v2) {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }
    template <typename T> inline T dot(const vector4<T>& v1, const vector4<T> v2) {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }
    
    template <typename T> inline T absdot(const vector2<T>& v1, const vector2<T> v2) {
        return std::abs(dot(v1, v2));
    }
    template <typename T> inline T absdot(const vector3<T>& v1, const vector3<T> v2) {
        return std::abs(dot(v1, v2));
    }
    template <typename T> inline T absdot(const vector4<T>& v1, const vector4<T> v2) {
        return std::abs(dot(v1, v2));
    }
    
    template <typename T> inline vector3<T> cross(const vector3<T>& v, const vector2<T>& v2) {
        T v1x = v1.x, v1y = v1.y, v1z = v1.z,
        v2x = v2.x, v2y = v2.y, v2z = v2.z;
        return vector3<T>(v1y * v2z - v1z * v2y,
                          v1z * v2x - v1x * v2z,
                          v1x * v2y - v1y * v2x);
    }
    
    template <typename T> inline vector2<T> normalize(const vector2<T>& v) { return v / v.length(); }
    template <typename T> inline vector3<T> normalize(const vector3<T>& v) { return v / v.length(); }
    template <typename T> inline vector4<T> normalize(const vector4<T>& v) { return v / v.length(); }
    
    template <typename T> T minComp(const vector2<T>& v) { return std::min(v.x, v.y); }
    template <typename T> T minComp(const vector3<T>& v) { return std::min(v.x, std::min(v.y, v.z)); }
    template <typename T> T minComp(const vector4<T>& v) { return std::min(v.x, std::min(v.y, std::min(v.z, v.w)); }
    template <typename T> T maxComp(const vector2<T>& v) { return std::max(v.x, v.y); }
    template <typename T> T maxComp(const vector3<T>& v) { return std::max(v.x, std::max(v.y, v.z)); }
    template <typename T> T maxComp(const vector4<T>& v) { return std::max(v.x, std::max(v.y, std::max(v.z, v.w)); }
                                                                                                                                                  
    template <typename T> vector2<T> min(const vector2<T>& v1, const vector2<T>& v2) {
      return vector2<T>(std::min(v1.x, v2.x), std::min(v1.y, v2.y));
    }
    template <typename T> vector3<T> min(const vector3<T>& v1, const vector3<T>& v2) {
      return vector3<T>(std::min(v1.x, v2.x), std::min(v1.y, v2.y), std::min(v1.z, v2.z));
    }
    template <typename T> vector4<T> min(const vector4<T>& v1, const vector4<T>& v2) {
      return vector4<T>(std::min(v1.x, v2.x), std::min(v1.y, v2.y), std::min(v1.z, v2.z), std::min(v1.w, v2.w));
    }
    template <typename T> vector2<T> max(const vector2<T>& v1, const vector2<T>& v2) {
      return vector2<T>(std::max(v1.x, v2.x), std::max(v1.y, v2.y));
    }
    template <typename T> vector3<T> max(const vector3<T>& v1, const vector3<T>& v2) {
      return vector3<T>(std::max(v1.x, v2.x), std::max(v1.y, v2.y), std::max(v1.z, v2.z));
    }
    template <typename T> vector4<T> max(const vector4<T>& v1, const vector4<T>& v2) {
      return vector4<T>(std::max(v1.x, v2.x), std::max(v1.y, v2.y), std::max(v1.z, v2.z), std::max(v1.w, v2.w));
    }

    template <typename T> vector2<T> permute(const vector2<T>& v, int x, int y) {
      return vector2<T>(v[x], v[y]);
    }
    template <typename T> vector3<T> permute(const vector2<T>& v, int x, int y, int z) {
      return vector2<T>(v[x], v[y], v[z]);
    }
    template <typename T> vector4<T> permute(const vector2<T>& v, int x, int y, int z, int w) {
      return vector2<T>(v[x], v[y], v[z], v[w]);
    }

    template <typename T> inline void makeBasis(const vector3<T>& v1, vector3<T>* v2, vector3<T>* v3) {
        if (std::abs(v1.x) > std::abs(v1.y))
            *v2 = vector3<T>(-v1.z, 0, v1.x) / std::sqrt(v1.x * v1.x + v1.z * v1.z);
        else
            *v2 = vector3<T>(0, v1.z, -v1.y) / std::sqrt(v1.x * v1.x + v1.z * v1.z);
        *v3 = cross(v1, *v2);
    }

    typedef vector2<Float> vector2f;
    typedef vector2<int> vector2i;
    typedef vector3<Float> vector3f;
    typedef vector3<int> vector3i;
    typedef vector4<Float> vector4f;
    typedef vector4<int> vector4i;

    /*====================
            Points
    =====================*/
        
    template <typename T> class point2 {
    public:
        T x, y;
        
        point2() { x = y = 0; }
        point2(T x, T y)
        : x(x), y(y) {
            Assert(valid());
        }
        point2<T>(const point2<T>& p) {
            Assert(p.valid());
            x = p.x; y = p.y;
        }
        point2<T>& operator=(const point2<T>& p) {
            Assert(p.valid());
            x = p.x; y = p.y;
            return *this;
        }
        explicit point2(const point3<T>& p)
        : x(p.x), y(p.y) {
            Assert(valid());
        }
        template <typename U> explicit point2(const point2<U>& p)
        : x((T)p.x), y((T)p.y) {
            Assert(valid());
        }
        template <typename U> explicit vector2<U>() const {
            return vector2<U>(x, y);
        }
        
        friend std::ostream& operator<<(std::ostream& os, const point2<T>& p) {
            os << "[" << p.x << ", " << p.y << "]";
            return os;
        }
        
        T operator[](int i) const {
            Assert(i >= 0 && i < 3);
            if (i == 0) return x;
            return y;
        }
        
        T& operator[](int i) {
            Assert(i >= 0 && i < 3);
            if (i == 0) return x;
            return y;
        }
        
        bool operator==(const point2<T>& p) const {
            return x == p.x && y == p.y;
        }
        bool operator!=(const point2<T>& p) const {
            return x != p.x || y != p.y;
        }
        
        point2<T> operator+(const vector2<T>& v) const {
            return point2<T>(x + v.x, y + v.y);
        }
        point2<T>& operator+=(const vector2<T>& v) {
            x += v.x; y += v.y;
            return *this;
        }
        vector2<T> operator-(const point2<T>& p) const {
            return vector2<T>(x - p.x, y - p.y);
        }
        point2<T> operator-(const vector2<T>& v) const {
            return point2<T>(x - v.x, y - v.y);
        }
        point2<T>& operator-=(const vector2<T>& v) {
            x -= v.x; y -= v.y;
            return *this;
        }
        
        bool valid() const {
            return !std::isnan(x) && !std::isnan(y)
        }
    }
                                                                           
    template <typename T> class point3 {
    public:
        T x, y, z;
        
        point3() { x = y = z = 0; }
        point3(T x, T y, T z)
        : x(x), y(y), z(z) {
            Assert(valid());
        }
        point3<T>(const point3<T>& p) {
            Assert(p.valid());
            x = p.x; y = p.y; z = p.z;
        }
        point3<T>& operator=(const point3<T>& p) {
            Assert(p.valid());
            x = p.x; y = p.y; z = p.z;
            return *this;
        }
        template <typename U> explicit point3(const point3<U>& p)
        : x((T)p.x), y((T)p.y), z((T)p.z) {
            Assert(valid());
        }
        template <typename U> explicit vector3<U>() const {
            return vector3<U>(x, y, z);
        }
        
        friend std::ostream& operator<<(std::ostream& os, const point3<T>& p) {
            os << "[" << p.x << ", " << p.y << ", " << p.z << "]";
            return os;
        }
        
        T operator[](int i) const {
            Assert(i >= 0 && i < 3);
            if (i == 0) return x;
            if (i == 1) return y;
            return z;
        }
        
        T& operator[](int i) {
            Assert(i >= 0 && i < 3);
            if (i == 0) return x;
            if (i == 0) return y;
            return z;
        }
        
        bool operator==(const point3<T>& p) const {
            return x == p.x && y == p.y && z == p.z;
        }
        bool operator!=(const point3<T>& p) const {
            return x != p.x || y != p.y || z != p.z;
        }
        
        point3<T> operator+(const vector3<T>& v) const {
            return point2<T>(x + v.x, y + v.y, z + v.z);
        }
        point3<T>& operator+=(const vector3<T>& v) {
            x += v.x; y += v.y; z += v.z;
            return *this;
        }
        point3<T> operator+(const point3<T>& p) const {
            return point2<T>(x + p.x, y + p.y, z + p.z);
        }
        point3<T>& operator+=(const point3<T>& p) {
            x += p.x; y += p.y; z += p.z;
            return *this;
        }
        vector3<T> operator-(const point3<T>& p) const {
            return vector3<T>(x - p.x, y - p.y, z - p.z);
        }
        point3<T> operator-(const vector3<T>& v) const {
            return point3<T>(x - v.x, y - v.y, z - v.z);
        }
        point3<T>& operator-=(const vector3<T>& v) {
            x -= v.x; y -= v.y; z -= v.z;
            return *this;
        }
        
        bool valid() const {
            return !std::isnan(x) && !std::isnan(y) && !std::isnan(z)
        }
    }
                                                                           
    template <typename T> inline Float distance(const point2<T>& p1, const point2<T>& p2) {
        return (p1 - p2).length();
    }
    template <typename T> inline Float squaredDistance(const point2<T>& p1, const point2<T>& p2) {
        return (p1 - p2).squaredLength();
    }
    template <typename T> inline Float distance(const point3<T>& p1, const point3<T>& p2) {
        return (p1 - p2).length();
    }
    template <typename T> inline Float squaredDistance(const point3<T>& p1, const point3<T>& p2) {
        return (p1 - p2).squaredLength();
    }
                                                                           
    template <typename T> point2<T> lerp(Float t, const point2<T>& p1, const point2<T>& p2) {
        return (1 - t) * p1 + t * p2;
    }
    template <typename T> point3<T> lerp(Float t, const point3<T>& p1, const point3<T>& p2) {
        return (1 - t) * p1 + t * p2;
    }
                                                                           
    template <typename T> point2<T> min(const point2<T>& p1, const point2<T>& p2) {
        return point2<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y));
    }
    template <typename T> point2<T> max(const point2<T>& p1, const point2<T>& p2) {
        return point2<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y));
    }
    template <typename T> point3<T> min(const point3<T>& p1, const point3<T>& p2) {
        return point3<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y), std::min(p1.z, p2.z));
    }
    template <typename T> point3<T> max(const point3<T>& p1, const point3<T>& p2) {
        return point2<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y), std::max(p1.z, p2.z));
    }
                                                                           
    template <typename T> point2<T> floor(const point2<T>& p) {
        return point2<T>(std::floor(x), std::floor(y));
    }
    template <typename T> point2<T> ceil(const point2<T>& p) {
        return point2<T>(std::ceil(x), std::ceil(y));
    }
    template <typename T> point2<T> abs(const point2<T>& p) {
        return point2<T>(std::abs(x), std::abs(y));
    }
    template <typename T> point3<T> floor(const point3<T>& p) {
        return point3<T>(std::floor(x), std::floor(y), std::floor(z));
    }
    template <typename T> point3<T> ceil(const point3<T>& p) {
        return point3<T>(std::ceil(x), std::ceil(y), std::ceil(z));
    }
    template <typename T> point3<T> abs(const point3<T>& p) {
        return point3<T>(std::abs(x), std::abs(y), std::abs(z));
    }
                                                                           
    template <typename T> point2<T> permute(const point2<T>& p, int x, int y) {
        return point2<T>(p[x], p[y]);
    }
    template <typename T> point3<T> permute(const point3<T>& p, int x, int y, int z) {
        return point3<T>(p[x], p[y], p[z]);
    }
                                                                           
    typedef point2<Float> point2f;
    typedef point2<int> point2i;
    typedef point3<Float> point3f;
    typedef point3<int> point3i;
                                                                           
    /*====================
            Normals
     =====================*/
        
    template <typename T> class normal3 {
    public:
        T x, y, z;
        
        normal3() { x = y = z = 0; }
        normal3(T x, T y, T z) : x(x), y(y), z(z) {
            Assert(valid());
        }
        normal3<T>(const normal3<T>& n) {
            Assert(n.valid());
            x = n.x; y = n.y; z = n.z;
        }
        normal3<T>& operator=(const normal3<T>& n) {
            Assert(n.valid());
            x = n.x; y = n.y; z = n.z;
            return *this;
        }
        explicit normal3<T>(const vector3<T>& v) : x(v.x), y(v.y), z(v.z) {
            Assert(v.valid());
        }
        
        friend std::ostream& operator<<(std::ostream& os, const normal3<T>& n) {
            os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
            return os;
        }
        
        T operator[](int i) const {
            Assert(i >= 0 && i < 3);
            if (i == 0) return x;
            if (i == 1) return y;
            return z;
        }
        
        T& operator[](int i) {
            Assert(i >= 0 && i < 3);
            if (i == 0) return x;
            if (i == 0) return y;
            return z;
        }
        
        bool operator==(const normal3<T>& n) const {
            return x == n.x && y == n.y && z == n.z;
        }
        bool operator!=(const normal3<T>& n) const {
            return x != n.x || y != n.y || z != n.z;
        }
        
        normal3<T> operator-() const {
            return normal3<T>(-x, -y, -z);
        }
        
        normal3<T> operator+(const normal3<T>& n) {
            return normal3<T>(x + n.x, y + n.y, z + n.z);
        }
        normal3<T> operator+=(const normal3<T>& n) {
            x += n.x; y += n.y; z += n.z;
            return *this;
        }
        normal3<T> operator-(const normal3<T>& n) {
            return normal3<T>(x - n.x, y - n.y, z - n.z);
        }
        normal3<T> operator-=(const normal3<T>& n) {
            x -= n.x; y -= n.y; z -= n.z;
            return *this;
        }
        normal3<T> operator*(T s) {
            return normal3<T>(x * s, y * s, z * s);
        }
        normal3<T> operator*=(T s) {
            x *= s; y *= s; z *= s;
            return *this;
        }
        normal3<T> operator/(T s) {
            Assert(s != 0);
            Float inv = (Float)1 / s;
            return normal3<T>(x * inv, y * inv, z * inv);
        }
        normal3<T> operator/=(T s) {
            Assert(s != 0);
            Float inv = (Float)1 / s;
            x *= inv; y *= inv; z *= inv;
            return *this;
        }
        
        Float squaredLength() const { return x * x + y * y + z * z; }
        Float length() const { return std::sqrt(squaredLength()); }
        
        bool valid() const {
            return !std::isnan(x) && !std::isnan(y) && !std::isnan(z)
        }
    }
        
    template <typename T> inline vector3<T>::vector3(const normal3<T>& n) : x(n.x), y(n.y), z(n.z) {
        Assert(n.valid());
    }
                                                                           
    template <typename T> inline normal3<T> faceForward(const normal3<T>& n, const vector3<T>& v) {
        return (dot(n, v) < 0) ? -n : n;
    }
        
    typedef normal3<Float> normal3f;
                                                                           
    /*====================
             Rays
     =====================*/
        
    class Ray {
    public:
        point3f o;
        vector3f d;
        mutable Float tMax;
        Float time;
        const Medium* medium;
        
        Ray() : tMax(infinity), time(0.f), medium(nullptr) { }
        Ray(const point3f& o, const vector3f& d, Float tMax = infinity, Float time = 0.f, const Medium* medium = nullptr)
        : o(o), d(d), tMax(tMax), time(time), medium(medium) { }
        
        point3f operator()(Float t) const { return o + d * t;}
        
        friend std::ostream& operator<<(std::ostream& os, const Ray& ray) {
            os << "[o=" << ray.o << ", d=" << ray.d << ", tMax=" << ray.tMax << ", time=" << ray.time << "]";
        }
        
        bool valid() {
            return o.valid() && d.valid() && !std::isnan(tMax);
        }
    };
                                                                           
    class RayDifferential : public Ray {
    public:
        bool hasDifferentials;
        point3f rxOrigin, ryOrigin;
        vector3f rxDirection, ryDirection;
        
        RayDifferential() { hasDifferentials = false; }
        RayDifferential(const point3f& o, const vector3f& d, Float tMax = infinity, Float time = 0.f, const Medium* medium = nullptr)
        : Ray(o, d, tMax, time, medium) { hasDifferentials = false; }
        RayDifferential(const Ray& ray) : Ray(ray) {
            hasDifferentials = false;
        }
        
        void scaleDifferentials(Float s) {
            rxOrigin = o + (rxOrigin - o) * s;
            ryOrigin = o + (ryOrigin - o) * s;
            rxDirection = d + (rxDirection - d) * s;
            ryDirection = d + (ryDirection - d) * s;
        }
        
        bool valid() {
            return Ray::valid() && (hasDiffentials ? (rxOrigin.valid() && ryOrigin.valid() && rxDirection.valid() && ryDirection.valid()) : true)
        }
    };
                                                                           
    /*====================
        Bounding Boxes
    =====================*/
        
    template <typename T> class bounds2 {
    public:
        point2<T> pMin, pMax;
        
        bounds2() {
            T min = std::numeric_limits<T>::lowest();
            T max = std::numeric_limits<T>::max();
            pMin = point2<T>(max, max);
            pMax = point2<T>(min, min);
        }
        bounds2(const point2<T>& p) : pMin(p), pMax(p) { }
        bounds2(const point2<T>& p1, const point2<T> p2)
        : pMin(std::min(p1.x, p2.x), std::min(p1.y, p2.y)),
          pMax(std::max(p1.x, p2.x), std::max(p1.y, p2.y)) { }
        
        const point2<T>& operator[](int i) const {
            Assert(i >= 0 && i < 2);
            if (i == 0) return point2<T>(pMin);
            return point2<T>(pMax);
        }
        point2<T>& operator[](int i) {
            Assert(i >= 0 && i < 2);
            if (i == 0) return pMin;
            return pMax;
        }
        
        point2<T> corner(int corner) const {
            return point2<T>((*this)[(corner & 1)].x,
                             (*this)[(corner & 2) ? 1 : 0].y);
        }
        
        vector2<T> diagonal() const { return pMax - pMin; }
        
        T surfaceArea() const {
            vector2<T> d = diagonal();
            return 2 * (d.x * d.y);
        }
        
        int maximumExtent() const {
            vector2<T> d = diagonal();
            return d.x > d.y ? 0 : 1;
        }
        
        point2<T> lerp(const point2f& t) const {
            return point2<T>(::lerp(t.x, pMin.x, pMax.x),
                             ::lerp(t.y, pMin.y, pMax.y));
        }
        
        vector2<T> offset(const point2<T>& p) const {
            vector2<T> o = p - pMin;
            if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
            if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
            return o;
        }
        
        void boundingSphere(point2<T>* center, Float* radius) {
            *center = (pMin + pMax) / 2;
            *radius = inside(*center, *this) ? distance(*center, pMax) : 0;
        }
    }

    template <typename T> class bounds3 {
    public:
        point3<T> pMin, pMax;
        
        bounds3() {
            T min = std::numeric_limits<T>::lowest();
            T max = std::numeric_limits<T>::max();
            pMin = point3<T>(max, max, max);
            pMax = point3<T>(min, min, min);
        }
        bounds3(const point3<T>& p) : pMin(p), pMax(p) { }
        bounds3(const point3<T>& p1, const point3<T> p2)
        : pMin(std::min(p1.x, p2.x), std::min(p1.y, p2.y), std::min(p1.z, p2.z)),
          pMax(std::max(p1.x, p2.x), std::max(p1.y, p2.y), std::max(p1.z, p2.z)) { }
        
        const point3<T>& operator[](int i) const {
            Assert(i >= 0 && i < 2);
            if (i == 0) return point3<T>(pMin);
            return point3<T>(pMax);
        }
        point3<T>& operator[](int i) {
            Assert(i >= 0 && i < 2);
            if (i == 0) return pMin;
            return pMax;
        }
        
        point3<T> corner(int corner) const {
            return point3<T>((*this)[(corner & 1)].x,
                             (*this)[(corner & 2) ? 1 : 0].y,
                             (*this)[(corner & 4) ? 1 : 0].z);
        }
        
        vector3<T> diagonal() const { return pMax - pMin; }
        
        T surfaceArea() const {
            vector3<T> d = diagonal();
            return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
        }
        
        T volume() const {
            vector3<T> d = diagonal();
            return d.x * d.y * d.z;
        }
        
        int maximumExtent() const {
            vector3<T> d = diagonal();
            if (d.x > d.y && d.x > d.z)
                return 0;
            else if (d.y > d.z)
                return 1;
            else return 0;
        }
        
        point3<T> lerp(const point3f& t) const {
            return point3<T>(::lerp(t.x, pMin.x, pMax.x),
                             ::lerp(t.y, pMin.y, pMax.y),
                             ::lerp(t.z, pMin.z, pMax.z));
        }
        
        vector3<T> offset(const point3<T>& p) const {
            vector3<T> o = p - pMin;
            if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
            if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
            if (pMax.z > pMin.z) o.z /= pMax.z - pMin.z;
            return o;
        }
        
        void boundingSphere(point3<T>* center, Float* radius) {
            *center = (pMin + pMax) / 2;
            *radius = inside(*center, *this) ? distance(*center, pMax) : 0;
        }
    }
                                                                           
    template <typename T> bounds2<T> union(const bounds2<T>& b, const point2<T> &p) {
        return bounds2<T>(point2<T>(std::min(b.pMin.x, p.x),
                                    std::min(b.pMin.y, p.y)),
                          point2<T>(std::max(b.pMax.x, p.x),
                                    std::max(b.pMax.y, p.y)));
    }
    template <typename T> bounds2<T> union(const bounds3<T>& b1, const bounds3<T>& b2) {
        return bounds2<T>(point2<T>(std::min(b1.pMin.x, b2.pMin.x),
                                    std::min(b1.pMin.y, b2.pMin.y)),
                          point2<T>(std::max(b1.pMax.x, b2.pMax.x),
                                    std::max(b1.pMax.y, b2.pMax.y)));
    }
    template <typename T> bounds3<T> union(const bounds3<T>& b, const point3<T> &p) {
        return bounds2<T>(point3<T>(std::min(b.pMin.x, p.x),
                                    std::min(b.pMin.y, p.y),
                                    std::min(b.pMin.z, p.z)),
                          point3<T>(std::max(b.pMax.x, p.x),
                                    std::max(b.pMax.y, p.y),
                                    std::max(b.pMax.z, p.z)));
    }
    template <typename T> bounds3<T> union(const bounds3<T>& b1, const bounds3<T>& b2) {
        return bounds2<T>(point3<T>(std::min(b1.pMin.x, b2.pMin.x),
                                    std::min(b1.pMin.y, b2.pMin.y),
                                    std::min(b1.pMin.z, b2.pMin.z)),
                          point3<T>(std::max(b1.pMax.x, b2.pMax.x),
                                    std::max(b1.pMax.y, b2.pMax.y),
                                    std::max(b1.pMax.z, b2.pMax.z)));
    }
                                                                           
    template <typename T> bounds2<T> intersection(const bounds2<T>& b1, const bounds2<T> b2) {
        return bounds2<T>(point2<T>(std::max(b1.pMin.x, b2.pMin.x),
                                    std::max(b1.pMin.y, b2.pMin.y)),
                          point2<T>(std::min(b1.pMax.x, b2.pMax.x),
                                    std::min(b1.pMax.y, b2.pMax.y)));
    }
    template <typename T> bounds3<T> intersection(const bounds3<T>& b1, const bounds3<T>& b2) {
        return bounds2<T>(point3<T>(std::max(b1.pMin.x, b2.pMin.x),
                                    std::max(b1.pMin.y, b2.pMin.y),
                                    std::max(b1.pMin.z, b2.pMin.z)),
                          point3<T>(std::min(b1.pMax.x, b2.pMax.x),
                                    std::min(b1.pMax.y, b2.pMax.y),
                                    std::min(b1.pMax.z, b2.pMax.z)));
    }
                                                                           
    template <typename T> bool overlaps(const bounds2<T>& b1, const bounds2<T>& b2) {
        bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
        bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
        return x && y;
    }
    template <typename T> bool overlaps(const bounds3<T>& b1, const bounds3<T>& b2) {
        bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
        bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
        bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
        return x && y && z;
    }
                                                                           
    template <typename T> bool inside(const point2<T>& p, const bounds2<T>& b) {
        return (p.x >= b.pMin.x && p.x <= b.pMax.x &&
                p.y >= b.pMin.y && p.y <= b.pMax.y);
    }
    template <typename T> bool inside(const point3<T>& p, const bounds3<T>& b) {
        return (p.x >= b.pMin.x && p.x <= b.pMax.x &&
                p.y >= b.pMin.y && p.y <= b.pMax.y &&
                p.z >= b.pMin.z && p.z <= b.pMax.z);
    }
    template <typename T> bool insideExclusive(const point2<T>& p, const bounds2<T>& b) {
        return (p.x >= b.pMin.x && p.x < b.pMax.x &&
                p.y >= b.pMin.y && p.y < b.pMax.y);
    }
    template <typename T> bool insideExclusive(const point3<T>& p, const bounds3<T>& b) {
        return (p.x >= b.pMin.x && p.x < b.pMax.x &&
                p.y >= b.pMin.y && p.y < b.pMax.y &&
                p.z >= b.pMin.z && p.z < b.pMax.z);
    }
                                                                           
    template <typename T, typename U> inline bounds2<T> expand(const bounds2<T>& b, U delta) {
        return bounds2<T>(b.pMin - vector2<T>(delta, delta),
                          b.pMax + vector2<T>(delta, delta));
    }
    template <typename T, typename U> inline bounds3<T> expand(const bounds3<T>& b, U delta) {
        return bounds3<T>(b.pMin - vector3<T>(delta, delta, delta),
                          b.pMax + vector3<T>(delta, delta, delta));
    }
        
    typedef bounds2<Float> bounds2f;
    typedef bounds2<int> bounds2i;
    typedef bounds3<Float> bounds3f;
    typedef bounds3<int> bounds3i;
                                                                           
    /*====================
         Transforms
    =====================*/
        
    class transform {
    public:
        transform() { }
        transform(const matrix4x4& m) : m(m), mInv(inverse(m)) { }
        transform(const matrix4x4& m, const matrix4x4& mInv) : m(m), mInv(mInv) { }
        transform(const Float mat[4][4]) {
            m = matrix4x4(mat);
            mInv = inverse(m);
        }
        
        friend transform inverse(const transform& t) {
            return transform(t.mInv, t.m);
        }
        friend transform transpose(const transform& t) {
            return transform(transpose(t.m), transpose(t.mInv));
        }
        
        transform translate(const vector3f& delta) {
            matrix4x4 m(1, 0, 0, delta.x,
                        0, 1, 0, delta.y,
                        0, 0, 1, delta.z,
                        0, 0, 0, 1);
            matrix4x4 mInv(1, 0, 0, -delta.x,
                           0, 1, 0, -delta.y,
                           0, 0, 1, -delta.z,
                           0, 0, 0, 1);
            return transform(m, mInv);
        }
        
        transform scale(Float x, Float y, Float z) {
            matrix4x4 m(x, 0, 0, 0,
                        0, y, 0, 0,
                        0, 0, z, 0,
                        0, 0, 0, 1);
            matrix4x4 mInv(1/x, 0, 0, 0,
                           0, 1/y, 0, 0,
                           0, 0, 1/z, 0,
                           0, 0, 0, 1);
            return transform(m, mInv);
        }
        
        transform rotateX(Float angle) {
            Float sinTheta = std::sin(rads(angle));
            Float cosTheta = std::cos(rads(angle));
            matrix4x4 m(1,        0,         0, 0,
                        0, cosTheta, -sinTheta, 0,
                        0, sinTheta,  cosTheta, 0,
                        0,        0,         0, 1);
            return transform(m, transpose(m));
        }
        
        transform rotateY(Float angle) {
            Float sinTheta = std::sin(rads(angle));
            Float cosTheta = std::cos(rads(angle));
            matrix4x4 m(cosTheta,  0, sinTheta, 0,
                        0,         1,        0, 0,
                        -sinTheta, 0, cosTheta, 0,
                        0,         0,         0, 1);
            return transform(m, transpose(m));
        }
        
        transform rotateZ(Float angle) {
            Float sinTheta = std::sin(rads(angle));
            Float cosTheta = std::cos(rads(angle));
            matrix4x4 m(cosTheta, -sinTheta, 0, 0,
                        sinTheta,  cosTheta, 0, 0,
                        0,                0, 1, 0,
                        0,                0, 0, 1);
            return transform(m, transpose(m));
        }
        
        transform rotate(Float angle, const vector3f& axis) {
            vector3f a = normalize(axis);
            Float sinTheta = std::sin(rads(angle));
            Float cosTheta = std::cos(rads(angle));
            matrix4x4 m;
            
            m.m[0][0] = a.x * a.x + (1 - a.x * a.x) * cosTheta;
            m.m[0][1] = a.x * a.y * (1 - cosTheta) - a.z * sinTheta;
            m.m[0][2] = a.x * a.z * (1 - cosTheta) + a.y * sinTheta;
            m.m[0][3] = 0;
            
            m.m[1][0] = a.x * a.y * (1 - cosTheta) + a.z * sinTheta;
            m.m[1][1] = a.y * a.y + (1 - a.y * a.y) * cosTheta;
            m.m[1][2] = a.y * a.z * (1 - cosTheta) - a.x * sinTheta;
            m.m[1][3] = 0;
            
            m.m[1][0] = a.x * a.z * (1 - cosTheta) - a.y * sinTheta;
            m.m[1][1] = a.y * a.z * (1 - cosTheta) + a.x * sinTheta;
            m.m[1][2] = a.z * a.z + (1 - a.z * a.z) * cosTheta;
            m.m[1][3] = 0;
            
            return transform(m, transpose(m));
        }
        
        transform lookat(const point3f& pos, const point3f& look, const vector3f& up) {
            matrix4x4 cameraToWorld;
            
            cameraToWorld.m[0][3] = pos.x;
            cameraToWorld.m[1][3] = pos.y;
            cameraToWorld.m[2][3] = pos.z;
            cameraToWorld.m[3][3] = 1;
            
            vector3f dir = normalize(look - pos);
            vector3f right = normalize(cross(normalize(up), dir));
            vector3f newUp = cross(dir, right);
            cameraToWorld.m[0][0] = right.x;
            cameraToWorld.m[1][0] = right.x;
            cameraToWorld.m[2][0] = right.x;
            cameraToWorld.m[3][0] = 0;
            cameraToWorld.m[0][1] = newUp.x;
            cameraToWorld.m[1][1] = newUp.x;
            cameraToWorld.m[2][1] = newUp.x;
            cameraToWorld.m[3][1] = 0;
            cameraToWorld.m[0][2] = dir.x;
            cameraToWorld.m[1][2] = dir.x;
            cameraToWorld.m[2][2] = dir.x;
            cameraToWorld.m[3][2] = 0;
            
            return transform(inverse(cameraToWorld), cameraToWorld);
        }
        
        bool hasScale() const {
            Float la2 = (*this)(vector3f(1, 0, 0)).squaredLength();
            Float lb2 = (*this)(vector3f(0, 1, 0)).squaredLength();
            Float lc2 = (*this)(vector3f(0, 0, 1)).squaredLength();
#define NOT_ONE(x) ((x) < .999f || (x) > 1.001f)
            return (NOT_ONE(la2) || NOT_ONE(lb2) || NOT_ONE(lc2));
#undef NOT_ONE
        }
        
    private:
        matrix4x4 m, mInv;
    }
                                                                           
    template <typename T> inline point3<T> transform::operator()(const point3<T>& p) const {
        T x = p.x, y = p.y, z = p.z;
        T xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
        T yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
        T zp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
        T wp = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];
        if (wp == 1) return point3<T>(xp, yp, zp);
        else return point3<T>(xp, yp, zp) / wp;
    }
                                                                           
    template <typename T> inline vector3<T> transform::operator()(const vector3<T>& v) const {
        T x = v.x, y = v.y, z = v.z;
        return vector3<T>(m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z,
                          m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z,
                          m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z);
    }
                                                                           
    template <typename T> inline normal3<T> transform::operator()(const normal3<T>& n) const {
        T x = n.x, y = n.y, z = n.z;
        return vector3<T>(mInv.m[0][0] * x + mInv.m[1][0] * y + mInv.m[2][0] * z,
                          mInv.m[0][1] * x + mInv.m[1][1] * y + mInv.m[2][1] * z,
                          mInv.m[0][2] * x + mInv.m[1][2] * y + mInv.m[2][2] * z);
    }
                                                                           
    inline ray transform::operator()(const Ray& n) const {
        point3f o = (*this)(r.o);
        vector3f d = (*this)(r.d);
        return Ray(o, d, r.tMax, r.time, r.medium);
    }
                                                                           
    bounds3f transform::operator()(const transform3f& b) const {
        const transform& m = *this;
        bounds3f ret(m(point3f(b.pMin.x, b.pMin.y, b.pMin.z)));
        ret = union(ret, m(point3f(b.pMax.x, b.pMin.y, b.pMin.z)));
        ret = union(ret, m(point3f(b.pMin.x, b.pMax.y, b.pMin.z)));
        ret = union(ret, m(point3f(b.pMin.x, b.pMin.y, b.pMax.z)));
        ret = union(ret, m(point3f(b.pMin.x, b.pMax.y, b.pMax.z)));
        ret = union(ret, m(point3f(b.pMax.x, b.pMax.y, b.pMin.z)));
        ret = union(ret, m(point3f(b.pMax.x, b.pMin.y, b.pMax.z)));
        ret = union(ret, m(point3f(b.pMax.x, b.pMax.y, b.pMax.z)));
        return ret;
    }
                                                                           
    transform transform::operator*(const transform& t2) const {
        return transform(m * t2.m, t2.mInv * mInv);
    }
                                                                           
    bool transform::swapsHandedness() const {
        Float det = m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]) -
                    m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0]) +
                    m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]);
        return det < 0;
    }
};
