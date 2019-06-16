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
	/* Vectors */
	template<typename T>
	class vector2 {
	public:
		T x, y;

		vector2() { x = y = 0; }
		vector2(T x, T y) : x(x), y(y) {
			Assert(valid());
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

	typedef vector2<Float> vector2f;
	typedef vector2<int> vector2i;
	typedef vector3<Float> vector3f;
	typedef vector3<int> vector3i;
	typedef vector4<Float> vector4f;
	typedef vector4<int> vector4i;
};