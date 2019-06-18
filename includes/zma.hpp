#pragma once

#include <cassert>
#include <cstdio>
#include <cmath>

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
    
    static constexpr Float infinity = std::numeric_limits<Float>::infinity();
    
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
        
    
};
