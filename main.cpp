/*
 *		 光线追踪（Ray Tracing）
 *	<作者> ：Margoo
 *	<邮箱> ：1683691371@qq.com
 */
#include <algorithm>
#include <conio.h>
#include <functional>
#include <graphics.h>
#include <intrin.h>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <xmmintrin.h>

/*
 * FloatX 类是对于 SSE 操作的一个封装，用来替换 float 的使用
 */
class FloatX
{
  public:
	__m128 SSEData;

  public:
	FloatX()
	{
		SSEData = _mm_set_ps1(0);
	}
	FloatX(const float &Data)
	{
		SSEData = _mm_set_ps1(Data);
	}
	FloatX(const __m128 &Data)
	{
		SSEData = Data;
	}
	FloatX(const FloatX &Object)
	{
		SSEData = Object.SSEData;
	}

	FloatX operator-() const
	{
		float NegaDouble = 0;
		auto NegaNum = _mm_set_ps1(NegaDouble);

		__m128 Result = _mm_sub_ps(NegaNum, SSEData);

		return FloatX(Result);
	}
	FloatX &operator+=(const FloatX &Value)
	{
		SSEData = _mm_add_ps(SSEData, Value.SSEData);

		return *this;
	}
	FloatX &operator*=(const FloatX &Value)
	{
		SSEData = _mm_mul_ps(SSEData, Value.SSEData);

		return *this;
	}
	FloatX &operator/=(const FloatX &Value)
	{
		SSEData = _mm_div_ps(SSEData, Value.SSEData);

		return *this;
	}

	void GetFloat(float *FloatData)
	{
		*FloatData = SSEData.m128_f32[0];
	}
	operator double()
	{
		return SSEData.m128_f32[0];
	}

	static FloatX Cos(FloatX X)
	{
		return _mm_cos_ps(X.SSEData);
	}
	static FloatX Sin(FloatX X)
	{
		return _mm_sin_ps(X.SSEData);
	}
	static FloatX Tan(FloatX X)
	{
		return _mm_tan_ps(X.SSEData);
	}

	static FloatX Sqrt(const FloatX &M128Object)
	{
		auto Result = _mm_sqrt_ps(M128Object.SSEData);

		return FloatX(Result);
	}
	static FloatX Pow(const FloatX &Value, const FloatX &Power)
	{
		auto Result = _mm_pow_ps(Value.SSEData, Power.SSEData);

		return FloatX(Result);
	}
	static FloatX Min(const FloatX &Left, const FloatX &Right)
	{
		return _mm_min_ps(Left.SSEData, Right.SSEData);
	}
	static FloatX Max(const FloatX &Left, const FloatX &Right)
	{
		return _mm_max_ps(Left.SSEData, Right.SSEData);
	}

	inline friend FloatX operator+(const FloatX &Left, const FloatX &Right)
	{
		return _mm_add_ps(Left.SSEData, Right.SSEData);
	}
	inline friend FloatX operator-(const FloatX &Left, const FloatX &Right)
	{
		return _mm_sub_ps(Left.SSEData, Right.SSEData);
	}
	inline friend FloatX operator*(const FloatX &Left, const FloatX &Right)
	{
		return _mm_mul_ps(Left.SSEData, Right.SSEData);
	}
	inline friend FloatX operator/(const FloatX &Left, const FloatX &Right)
	{
		return _mm_div_ps(Left.SSEData, Right.SSEData);
	}

	inline friend FloatX operator+(const FloatX &Left, const float &Right)
	{
		auto RightValue = _mm_set_ps1(Right);

		return _mm_add_ps(Left.SSEData, RightValue);
	}
	inline friend FloatX operator-(const FloatX &Left, const float &Right)
	{
		auto RightValue = _mm_set_ps1(Right);

		return _mm_sub_ps(Left.SSEData, RightValue);
	}
	inline friend bool operator>(FloatX &Left, FloatX &Right)
	{
		return (float)Left > (float)Right;
	}
	inline friend bool operator<(FloatX &Left, FloatX &Right)
	{
		return (float)Left < (float)Right;
	}
	inline friend bool operator>=(FloatX &Left, FloatX &Right)
	{
		return (float)Left >= (float)Right;
	}
	inline friend bool operator<=(FloatX &Left, FloatX &Right)
	{
		return (float)Left <= (float)Right;
	}
	inline friend bool operator==(FloatX &Left, FloatX &Right)
	{
		return (float)Left == (float)Right;
	}
	inline friend FloatX operator/(const FloatX &Left, const float &Right)
	{
		auto RightValue = _mm_set_ps1(Right);

		return _mm_div_ps(Left.SSEData, RightValue);
	}
	inline friend FloatX operator+(const float &Left, const FloatX &Right)
	{
		return Right + Left;
	}
	inline friend FloatX operator-(const float &Left, const FloatX &Right)
	{
		auto LeftValue = _mm_set_ps1(Left);

		return _mm_sub_ps(LeftValue, Right.SSEData);
	}
	inline friend FloatX operator/(const float &Left, const FloatX &Right)
	{
		auto LeftValue = _mm_set_ps1(Left);

		return _mm_div_ps(LeftValue, Right.SSEData);
	}
};

/*
 * 定义 FloatX 的字面量，方便使用
 */
FloatX operator"" fx(long double Value)
{
	return FloatX(static_cast<float>(Value));
}

/*
 * 常用常量定义
 */
const FloatX Inf = std::numeric_limits<float>::infinity();
const FloatX PI = 3.1415926535897932385fx;

/*
 * 角度转为弧度
 */
FloatX DegreesToRadians(FloatX Degrees)
{
	return Degrees * PI / 180.fx;
}

float Random()
{
	return rand() / (RAND_MAX + 1.f);
}
FloatX Random(FloatX Max, FloatX Min)
{
	return Min + (Max - Min) * FloatX(Random());
}
FloatX Range(FloatX X, FloatX Min, FloatX Max)
{
	if (X < Min)
	{
		return Min;
	}
	if (X > Max)
	{
		return Max;
	}

	return X;
}

/*
 * 三维向量类 Vector3 （同时该类也充当了三维点和颜色）
 */
typedef class Vector3
{
  public:
	Vector3() : Element{0.f, 0.f, 0.f}
	{
	}
	Vector3(FloatX X, FloatX Y, FloatX Z) : Element{X, Y, Z}
	{
	}

	FloatX GetX() const
	{
		return Element[0];
	}
	FloatX GetY() const
	{
		return Element[1];
	}
	FloatX GetZ() const
	{
		return Element[2];
	}

	Vector3 operator-() const
	{
		return Vector3(-Element[0], -Element[1], -Element[2]);
	}
	FloatX operator[](const unsigned int &Position) const
	{
		if (Position >= 3)
		{
			abort();

			return Element[0];
		}
		else
		{
			return Element[Position];
		}
	}
	FloatX &operator[](const unsigned int &Position)
	{
		if (Position >= 3)
		{
			abort();

			return Element[0];
		}
		else
		{
			return Element[Position];
		}
	}
	Vector3 &operator+=(const Vector3 &Value)
	{
		Element[0] += Value.Element[0];
		Element[1] += Value.Element[1];
		Element[2] += Value.Element[2];

		return *this;
	}
	Vector3 &operator*=(const FloatX &Value)
	{
		Element[0] *= Value;
		Element[1] *= Value;
		Element[2] *= Value;

		return *this;
	}
	Vector3 &operator/=(const FloatX &Value)
	{
		return *this *= 1 / Value;
	}

	FloatX LengthSquared() const
	{
		return Element[0] * Element[0] + Element[1] * Element[1] + Element[2] * Element[2];
	}
	FloatX Length() const
	{
		return FloatX::Sqrt(LengthSquared());
	}

	inline friend Vector3 operator+(const Vector3 &Left, const Vector3 &Right)
	{
		return Vector3(Left.GetX() + Right.GetX(), Left.GetY() + Right.GetY(), Left.GetZ() + Right.GetZ());
	}
	inline friend Vector3 operator-(const Vector3 &Left, const Vector3 &Right)
	{
		return Vector3(Left.GetX() - Right.GetX(), Left.GetY() - Right.GetY(), Left.GetZ() - Right.GetZ());
	}
	inline friend Vector3 operator*(const Vector3 &Left, const Vector3 &Right)
	{
		return Vector3(Left.GetX() * Right.GetX(), Left.GetY() * Right.GetY(), Left.GetZ() * Right.GetZ());
	}
	inline friend Vector3 operator*(const Vector3 &Left, const FloatX &Right)
	{
		return Vector3(Left.GetX() * Right, Left.GetY() * Right, Left.GetZ() * Right);
	}
	inline friend Vector3 operator*(const FloatX &Right, const Vector3 &Left)
	{
		return Vector3(Left.GetX() * Right, Left.GetY() * Right, Left.GetZ() * Right);
	}
	inline friend Vector3 operator/(const Vector3 &Left, const FloatX &Right)
	{
		return Vector3(Left.GetX() / Right, Left.GetY() / Right, Left.GetZ() / Right);
	}

	static Vector3 RandomInUnitSphere()
	{
		auto A = Random(0.fx, 2.fx * PI);
		auto Z = Random(-1.fx, 1.fx);
		auto R = FloatX::Sqrt(1.fx - Z * Z);

		return Vector3(R * FloatX::Cos(A), R * FloatX::Sin(A), Z);
	}

  private:
	FloatX Element[3];
} Point3, Color;

/*
 * 向量运算函数
 */
inline FloatX Vector3Dot(const Vector3 &Left, const Vector3 &Right)
{
	return Left[0] * Right[0] + Left[1] * Right[1] + Left[2] * Right[2];
}
inline Vector3 Vector3Cross(const Vector3 &Left, const Vector3 &Right)
{
	return Vector3(Left[1] * Right[2] - Left[2] * Right[1], Left[2] * Right[0] - Left[0] * Right[2],
				   Left[0] * Right[1] - Left[1] * Right[0]);
}
inline Vector3 UnitVector(Vector3 Vector)
{
	return Vector / Vector.Length();
}

/*
 * 判断法向量是否在同一个半球。
 */
Vector3 RandomInHemisphere(const Vector3 &NormalSurface)
{
	Vector3 InUnitSphere = Vector3::RandomInUnitSphere();
	if (Vector3Dot(InUnitSphere, NormalSurface) > 0.fx)
	{
		return InUnitSphere;
	}
	else
	{
		return -InUnitSphere;
	}
}

/*
 * 光线类
 */
class Ray
{
  public:
	Ray()
	{
	}
	Ray(const Point3 &OriginPoint, const Vector3 &LightDirection) : LightOrigin(OriginPoint), Direction(LightDirection)
	{
	}

	Point3 OriginPoint() const
	{
		return LightOrigin;
	}
	Vector3 LightDirection() const
	{
		return Direction;
	}

	Point3 LightAt(const FloatX &T) const
	{
		return LightOrigin + T * Direction;
	}

	static Vector3 RandomRay()
	{
		return Vector3(Random(), Random(), Random());
	}
	static Vector3 RandomRay(FloatX Min, FloatX Max)
	{
		return Vector3(Random(Min, Max), Random(Min, Max), Random(Min, Max));
	}

  public:
	// 光线出发点
	Point3 LightOrigin;
	// 光线方向
	Vector3 Direction;
};

/*
 * BVH 优化
 */
class BVHBox
{
  public:
	BVHBox()
	{
	}
	BVHBox(const Vector3 &Min, const Vector3 &Max)
	{
		BoxMin = Min;
		BoxMax = Max;
	}

	bool Hit(const Ray &RayInstance, FloatX TMin, FloatX TMax) const
	{
		for (auto Count = 0; Count < 3; ++Count)
		{
			auto InvD = 1.fx / RayInstance.Direction[Count];
			auto T0 = (BoxMin[Count] - RayInstance.LightOrigin[Count]) * InvD;
			auto T1 = (BoxMax[Count] - RayInstance.LightOrigin[Count]) * InvD;

			if (InvD < 0.fx)
			{
				auto Temp = T0;

				T0 = T1;
				T1 = Temp;
			}

			TMin = T0 > TMin ? T0 : TMin;
			TMax = T1 < TMax ? T1 : TMax;

			if (TMax <= TMin)
			{
				return false;
			}
		}

		return true;
	}

  public:
	Vector3 BoxMin;
	Vector3 BoxMax;
};

/*
 * 一个三维对象的材质基类
 */
class ObjectMaterial;

/*
 * 定义了光线求交所需数据
 */
struct HitData
{
	Vector3 NormalSurface;
	Vector3 RayData;
	FloatX T;
	FloatX U;
	FloatX V;
	bool FrontFace;
	ObjectMaterial *Material;

	void SetSurfaceNormal(const Ray &RayInstance, const Vector3 &OutsideNormal)
	{
		FrontFace = Vector3Dot(RayInstance.Direction, OutsideNormal) < 0;
		NormalSurface = FrontFace ? OutsideNormal : -OutsideNormal;
	}
};

/*
 * 材质基类
 */
class ObjectMaterial
{
  public:
	/*
	 * 用于实现光源的 Emitted，如果该物质本身不发光，返回纯黑
	 */
	virtual Vector3 Emitted(FloatX U, FloatX V, const Vector3 &RayData)
	{
		return Vector3(0.fx, 0.fx, 0.fx);
	}
	/*
	 * 材质的光散射
	 */
	virtual bool LightScatter(const Ray &RayIn, const HitData &Data, Vector3 &AttenuationLevel,
							  Ray &ScatterRay) const = 0;

  public:
	/*
	 * 光反射计算
	 */
	Vector3 Reflect(const Vector3 &RayIn, const Vector3 &NormalRay) const
	{
		return RayIn - 2.fx * Vector3Dot(RayIn, NormalRay) * NormalRay;
	}
};

/*
 * 材质的基类
 */
class Texture
{
  public:
	virtual Vector3 GetValue(FloatX U, FloatX V, const Vector3 &Point) const = 0;
};

/*
 * 固定颜色的材质
 */

class ConstantTexture : public Texture
{
  public:
	ConstantTexture()
	{
	}
	ConstantTexture(Vector3 TextureColor) : Color(TextureColor)
	{
	}
	Vector3 GetValue(FloatX U, FloatX V, const Vector3 &Point) const override
	{
		return Color;
	}

  public:
	Vector3 Color;
};
/*
 * 棋盘材质
 */
class CheckerTexture : public Texture
{
  public:
	CheckerTexture()
	{
	}
	CheckerTexture(Texture *Even, Texture *Odd) : EvenTexture(Even), OddTexture(Odd)
	{
	}
	Vector3 GetValue(FloatX U, FloatX V, const Vector3 &Point) const override
	{
		auto Sines =
			FloatX::Sin(10.fx * Point.GetX()) * FloatX::Sin(10.fx * Point.GetY()) * FloatX::Sin(10.fx * Point.GetZ());
		if (Sines < 0.fx)
		{
			return OddTexture->GetValue(U, V, Point);
		}
		else
		{
			return EvenTexture->GetValue(U, V, Point);
		}
	}

  public:
	Texture *EvenTexture;
	Texture *OddTexture;
};

/*
 * 漫反射材质
 */
class LambertianMaterial : public ObjectMaterial
{
  public:
	LambertianMaterial(Texture *MaterialTexture) : Texture(MaterialTexture)
	{
	}
	bool LightScatter(const Ray &RayIn, const HitData &Data, Vector3 &AttenuationLevel, Ray &ScatterRay) const override
	{
		Vector3 ScatterDirection = Data.NormalSurface + Vector3::RandomInUnitSphere();
		ScatterRay = Ray(Data.RayData, ScatterDirection);
		AttenuationLevel = Texture->GetValue(Data.U, Data.V, Data.RayData);

		return true;
	}

  public:
	Texture *Texture;
};

/*
 * 金属材质
 */
class MetalMaterial : public ObjectMaterial
{
  public:
	MetalMaterial(const Vector3 &Albedo, FloatX Fuzz) : MaterialAlbedo(Albedo), MetalFuzz(Fuzz < 1.fx ? Fuzz : 1.fx)
	{
	}
	bool LightScatter(const Ray &RayIn, const HitData &Data, Vector3 &AttenuationLevel, Ray &ScatterRay) const override
	{
		Vector3 ScatterDirection = Reflect(UnitVector(RayIn.Direction), Data.NormalSurface);
		ScatterRay = Ray(Data.RayData, ScatterDirection + MetalFuzz * Vector3::RandomInUnitSphere());
		AttenuationLevel = MaterialAlbedo;

		return Vector3Dot(ScatterRay.Direction, Data.NormalSurface) > 0;
	}

  public:
	FloatX MetalFuzz;
	Vector3 MaterialAlbedo;
};

/*
 * 绝缘体（一般是玻璃）材质
 */
class DielectricMaterial : public ObjectMaterial
{
  public:
	DielectricMaterial(FloatX RefractiveIndex) : MaterialRefractiveIndex(RefractiveIndex)
	{
	}
	bool LightScatter(const Ray &RayIn, const HitData &Data, Vector3 &AttenuationLevel, Ray &ScatterRay) const override
	{
		AttenuationLevel = Vector3(1.fx, 1.fx, 1.fx);
		FloatX EtaiOverEtat = (Data.FrontFace) ? (1.fx / MaterialRefractiveIndex) : (MaterialRefractiveIndex);

		Vector3 UnitDirection = UnitVector(RayIn.Direction);
		FloatX CosTheta = FloatX::Min(Vector3Dot(-UnitDirection, Data.NormalSurface), 1.fx);
		FloatX SinTheta = FloatX::Sqrt(1.fx - CosTheta * CosTheta);

		if (EtaiOverEtat * SinTheta > 1.fx)
		{
			Vector3 ReflectedRay = Reflect(UnitDirection, Data.NormalSurface);
			ScatterRay = Ray(Data.RayData, ReflectedRay);

			return true;
		}

		FloatX ReflectProb = Schlick(CosTheta, EtaiOverEtat);
		if (Random() < ReflectProb)
		{
			Vector3 ReflectedRay = Reflect(UnitDirection, Data.NormalSurface);
			ScatterRay = Ray(Data.RayData, ReflectedRay);

			return true;
		}

		Vector3 ReflectedRay = Refract(UnitDirection, Data.NormalSurface, EtaiOverEtat);
		ScatterRay = Ray(Data.RayData, ReflectedRay);

		return true;
	}

  public:
	Vector3 Refract(const Vector3 &UV, const Vector3 &RayIn, FloatX EtaiOverEtat) const
	{
		auto CosTheta = Vector3Dot(-UV, RayIn);

		Vector3 OutParallel = EtaiOverEtat * (UV + CosTheta * RayIn);
		Vector3 OutPerp = -FloatX::Sqrt(1.fx - OutParallel.LengthSquared()) * RayIn;

		return OutParallel + OutPerp;
	}
	/*
	 * Christophe Schlick 提出的折射率计算方法
	 */
	FloatX Schlick(FloatX Cos, FloatX RefractiveIndex) const
	{
		auto R0 = (1.fx - RefractiveIndex) / (1.fx + RefractiveIndex);
		R0 *= R0;

		return R0 + (1.fx - R0) * FloatX::Pow((1.fx - Cos), 5);
	}

  private:
	FloatX MaterialRefractiveIndex;
};

/*
 * 光源材质
 */
class DiffuseLight : public ObjectMaterial
{
  public:
	DiffuseLight(Vector3 Color) : LightColor(Color)
	{
	}

	bool LightScatter(const Ray &RayIn, const HitData &Data, Vector3 &AttenuationLevel, Ray &ScatterRay) const override
	{
		return false;
	}
	/*
	 * 光源则返回光源的颜色
	 */
	Vector3 Emitted(FloatX U, FloatX V, const Vector3 &RayData)
	{
		return LightColor;
	}

  public:
	Vector3 LightColor;
};

class HitTestBase
{
  public:
	virtual bool HitTest(const Ray &RayInstance, FloatX MinT, FloatX MaxT, HitData &Data) const = 0;
	// 用于 BVH 优化的盒型模型
	virtual bool HittingBox(FloatX T0, FloatX T1, BVHBox &Box) const = 0;

	BVHBox GetBox(const BVHBox &FirstBox, const BVHBox &SecondBox) const
	{
		Vector3 Min(FloatX::Min(FirstBox.BoxMin.GetX(), SecondBox.BoxMin.GetX()),
					FloatX::Min(FirstBox.BoxMin.GetY(), SecondBox.BoxMin.GetY()),
					FloatX::Min(FirstBox.BoxMin.GetZ(), SecondBox.BoxMin.GetZ()));
		Vector3 Max(FloatX::Max(FirstBox.BoxMax.GetX(), SecondBox.BoxMax.GetX()),
					FloatX::Max(FirstBox.BoxMax.GetY(), SecondBox.BoxMax.GetY()),
					FloatX::Max(FirstBox.BoxMax.GetZ(), SecondBox.BoxMax.GetZ()));

		return BVHBox(Min, Max);
	}
};

class SphereObject : public HitTestBase
{
  public:
	SphereObject() : CenterPoint(0.fx, 0.fx, 0.fx), Radius(0.fx)
	{
	}
	SphereObject(Vector3 Center, FloatX SphereRadius, ObjectMaterial *ObjectrMaterial)
		: CenterPoint(Center), Radius(SphereRadius), Material(ObjectrMaterial)
	{
	}

	bool HitTest(const Ray &RayInstance, FloatX MinT, FloatX MaxT, HitData &Data) const override
	{
		Vector3 RealtivlyPoint = RayInstance.LightOrigin - CenterPoint;

		auto A = RayInstance.Direction.LengthSquared();
		auto HalfB = Vector3Dot(RealtivlyPoint, RayInstance.Direction);
		auto C = RealtivlyPoint.LengthSquared() - Radius * Radius;
		auto Statment = HalfB * HalfB - A * C;

		if (Statment > 0)
		{
			auto Root = FloatX::Sqrt(Statment);
			auto TempValue = (-HalfB - Root) / A;

			if (TempValue < MaxT && TempValue > MinT)
			{
				Data.T = TempValue;
				Data.RayData = RayInstance.LightAt(Data.T);

				Vector3 OutwardSurface = (Data.RayData - CenterPoint) / Radius;

				Data.Material = Material;
				Data.SetSurfaceNormal(RayInstance, OutwardSurface);

				return true;
			}
			else
			{
				TempValue = (-HalfB + Root) / A;

				if (TempValue < MaxT && TempValue > MinT)
				{
					Data.T = TempValue;
					Data.RayData = RayInstance.LightAt(Data.T);

					Vector3 OutwardSurface = (Data.RayData - CenterPoint) / Radius;

					Data.Material = Material;
					Data.SetSurfaceNormal(RayInstance, OutwardSurface);

					return true;
				}
			}
		}

		return false;
	}
	bool HittingBox(FloatX T0, FloatX T1, BVHBox &Box) const override
	{
		Box = BVHBox(CenterPoint - Vector3(Radius, Radius, Radius), CenterPoint + Vector3(Radius, Radius, Radius));

		return true;
	}

  public:
	Vector3 CenterPoint;
	FloatX Radius;
	ObjectMaterial *Material;
};

typedef class ObjectList : public HitTestBase
{
  public:
	ObjectList()
	{
	}
	ObjectList(HitTestBase *Object)
	{
		Objects.push_back(Object);
	}

	void Clear()
	{
		Objects.clear();
	}
	void PushObject(HitTestBase *Object)
	{
		Objects.push_back(Object);
	}

	bool HitTest(const Ray &RayInstance, FloatX MinT, FloatX MaxT, HitData &Data) const override
	{
		HitData CacheData;
		bool AlreadyHit = false;
		FloatX ClosestHit = MaxT;

		for (auto &Object : Objects)
		{
			if (Object->HitTest(RayInstance, MinT, ClosestHit, CacheData))
			{
				AlreadyHit = true;
				ClosestHit = CacheData.T;
				Data = CacheData;
			}
		}

		return AlreadyHit;
	}
	// 用于 BVH 优化的盒型模型
	bool HittingBox(FloatX T0, FloatX T1, BVHBox &Box) const override
	{
		if (Objects.empty())
		{
			return false;
		}

		BVHBox CacheBox;
		bool FirstHit = true;

		for (auto &Object : Objects)
		{
			/*
			 * 当一个对象的碰撞想不在该范围的时候，其所有对象也必然不会处于 BVH 中
			 */
			if (!Object->HittingBox(T0, T1, Box))
			{
				return false;
			}

			CacheBox = FirstHit ? CacheBox : GetBox(Box, CacheBox);
			FirstHit = true;
		}

		return true;
	}

  public:
	std::vector<HitTestBase *> Objects;
} TraceWorld;

typedef class BVHRoot : public HitTestBase
{
  public:
	bool BoxCompare(const HitTestBase *Left, const HitTestBase *Righ, int Axis)
	{
		BVHBox BoxLeft;
		BVHBox BoxRight;

		return BoxLeft.BoxMin[Axis] < BoxRight.BoxMin[Axis];
	}
	bool BoxCompareX(const HitTestBase *Left, const HitTestBase *Right)
	{
		return BoxCompare(Left, Right, 0);
	}
	bool BoxCompareY(const HitTestBase *Left, const HitTestBase *Right)
	{
		return BoxCompare(Left, Right, 1);
	}
	bool BoxCompareZ(const HitTestBase *Left, const HitTestBase *Right)
	{
		return BoxCompare(Left, Right, 2);
	}

  public:
	BVHRoot()
	{
	}
	BVHRoot(std::vector<HitTestBase *> Objects, size_t Begin, size_t End, FloatX Time0, FloatX Time1)
	{
		int Axis = rand() % 2;
		auto Comparator =
			(Axis == 0)	  ? (std::bind(&BVHRoot::BoxCompareX, this, std::placeholders::_1, std::placeholders::_2))
			: (Axis == 1) ? (std::bind(&BVHRoot::BoxCompareY, this, std::placeholders::_1, std::placeholders::_2))
						  : (std::bind(&BVHRoot::BoxCompareZ, this, std::placeholders::_1, std::placeholders::_2));
		auto Range = End - Begin;

		if (Range == 1)
		{
			RightNode = Objects[Begin];
			LeftNode = RightNode;
		}
		else if (Range == 2)
		{
			if (Comparator(Objects[Begin], Objects[Begin + 1]))
			{
				LeftNode = Objects[Begin];
				RightNode = Objects[Begin + 1];
			}
			else
			{
				LeftNode = Objects[Begin + 1];
				RightNode = Objects[Begin];
			}
		}
		else
		{
			std::sort(Objects.begin() + Begin, Objects.begin() + End, Comparator);

			auto Middle = Begin + Range / 2;

			LeftNode = new BVHNode(Objects, Begin, Middle, Time0, Time1);
			RightNode = new BVHNode(Objects, Middle, End, Time0, Time1);
		}

		BVHBox LeftBox;
		BVHBox RightBox;
		LeftNode->HittingBox(Time0, Time1, LeftBox);
		RightNode->HittingBox(Time0, Time1, RightBox);

		Box = GetBox(LeftBox, RightBox);
	}
	BVHRoot(TraceWorld &World, FloatX Time0, FloatX Time1)
		: BVHRoot(World.Objects, 0, World.Objects.size(), Time0, Time1)
	{
	}

	bool HitTest(const Ray &RayInstance, FloatX TMin, FloatX TMax, HitData &Data) const override
	{
		if (!Box.Hit(RayInstance, TMin, TMax))
		{
			return false;
		}

		bool HitLeft = LeftNode->HitTest(RayInstance, TMin, TMax, Data);
		bool HitRight = RightNode->HitTest(RayInstance, TMin, HitLeft ? Data.T : TMax, Data);

		return HitLeft || HitRight;
	}
	bool HittingBox(FloatX T0, FloatX T1, BVHBox &HitBox) const override
	{
		HitBox = Box;

		return true;
	}

  public:
	HitTestBase *LeftNode;
	HitTestBase *RightNode;

	BVHBox Box;

} BVHNode;

class Camera
{
  public:
	Camera(Vector3 LookFrom, Vector3 LookAt, Vector3 VUP, FloatX VFov, FloatX Aspect)
	{
		OriginPoint = LookFrom;

		Vector3 U, V, W;

		auto Theta = DegreesToRadians(VFov);
		auto HalfHeight = FloatX::Tan(Theta / 2.fx);
		auto HalfWidth = Aspect * HalfHeight;

		W = UnitVector(LookFrom - LookAt);
		U = UnitVector(Vector3Cross(VUP, W));
		V = Vector3Cross(W, U);

		LowerLeftCorner = OriginPoint - HalfWidth * U - HalfHeight * V - W;

		Horizontal = 2.fx * HalfWidth * U;
		Vertical = 2.fx * HalfHeight * V;
	}

	Ray GetRay(FloatX U, FloatX V)
	{
		return Ray(OriginPoint, LowerLeftCorner + U * Horizontal + V * Vertical - OriginPoint);
	}

  public:
	Vector3 OriginPoint;
	Vector3 LowerLeftCorner;
	Vector3 Horizontal;
	Vector3 Vertical;
};

DWORD GetRawColor(const Vector3 &Vector, FloatX SamplePerPixel)
{
	auto Scale = 1.fx / SamplePerPixel;
	auto R = FloatX::Sqrt(Scale * Vector[0]);
	auto G = FloatX::Sqrt(Scale * Vector[1]);
	auto B = FloatX::Sqrt(Scale * Vector[2]);

	return RGB(Range(B, 0.fx, 0.999fx) * 256.fx, Range(G, 0.fx, 0.999fx) * 256.fx, Range(R, 0.fx, 0.999fx) * 256.fx);
}

/*
 * 实际上的渲染是由下到上地渲染，需要镜面反转渲染的点
 */
template <class Type> Type OppositeCoordinate(const Type &X, const Type &Y, const Type &Width, const Type &Height)
{
	return (Height - Y - 1) * Width + X;
}

/*
 * 渲染函数
 */
Vector3 RayColor(const Ray &RayObject, HitTestBase *World, int Depth)
{
	HitData Data;

	if (Depth <= 0)
	{
		return Vector3(0.fx, 0.fx, 0.fx);
	}

	if (!World->HitTest(RayObject, 0.001fx, Inf, Data))
	{
		return Vector3(0.fx, 0.fx, 0.fx);
	}

	Ray ScatteredRay;
	Vector3 Attenuation;
	Vector3 Emitted = Data.Material->Emitted(Data.U, Data.V, Data.RayData);

	if (!Data.Material->LightScatter(RayObject, Data, Attenuation, ScatteredRay))
	{
		return Emitted;
	}

	return Emitted + Attenuation * RayColor(ScatteredRay, World, Depth - 1);
}

TraceWorld World;
BVHRoot *BVHWorld;
Camera *WorldCamera;
HitTestBase *RenderWorld;

bool RenderStatus[48];

void Render(int StartX, int StartY, int EndX, int EndY, int Width, int Height, int MaxRenderDepth, int SamplePerPixel,
			const int &ThreadCount)
{
	auto Buffer = GetImageBuffer();

	for (int Y = EndY; Y >= StartY; --Y)
	{
		for (int X = StartX; X < EndX; ++X)
		{
			Vector3 Color;
			for (int SampleCount = 0; SampleCount < SamplePerPixel; ++SampleCount)
			{
				auto U = (float)(X + Random()) / Width;
				auto V = (float)(Y + Random()) / Height;

				auto RayInstance = WorldCamera->GetRay(U, V);
				Color += RayColor(RayInstance, RenderWorld, MaxRenderDepth);
			}

			Buffer[OppositeCoordinate(X, Y, Width, Height)] =
				GetRawColor(Color, FloatX(static_cast<float>(SamplePerPixel)));
		}
	}

	RenderStatus[ThreadCount] = true;
}
void StartThreadRender(const int &Width, const int &Height, const int &MaxRenderDepth, const int &SamplePerPixel)
{
	for (int Count = 0; Count < 48; ++Count)
	{
		RenderStatus[Count] = false;
	}

	auto Count = 0;

	for (int Y = Height - 1; Y >= 0; Y -= 80)
	{
		for (int X = 0; X < Width; X += 80)
		{
			if (Y - 80 >= 0)
			{
				std::thread RendThread(Render, X, Y - 80, X + 80, Y, Width, Height, MaxRenderDepth, SamplePerPixel,
									   Count);

				RendThread.detach();
			}
			else
			{
				std::thread RendThread(Render, X, Y - 79, X + 80, Y, Width, Height, MaxRenderDepth, SamplePerPixel,
									   Count);

				RendThread.detach();
			}

			++Count;
		}
	}

	while (true)
	{
		bool Flag = false;

		for (int Count = 0; Count < 48; ++Count)
		{
			if (!RenderStatus[Count])
			{
				Flag = true;
			}
		}

		if (!Flag)
		{
			return;
		}

		Sleep(4);
	}
}

void Room1()
{
	World.Clear();

	auto GroundMaterial = new LambertianMaterial(new ConstantTexture(Vector3(0.fx, 1.fx, 0.fx)));
	auto LightMaterial = new DiffuseLight(Vector3(0.fx, 1.fx, 0.fx));
	auto Glass = new DielectricMaterial(1.5);

	auto Ground = new SphereObject(Vector3(0.fx, -100.5fx, -1.fx), 100.fx, GroundMaterial);
	auto Light = new SphereObject(Vector3(0.fx, 0.fx, -1.fx), 0.4fx, LightMaterial);
	auto GlassObject = new SphereObject(Vector3(-1.fx, 0.fx, -1.fx), 0.5fx, Glass);

	World.PushObject(Ground);
	World.PushObject(Light);
	World.PushObject(GlassObject);
}
void Room2()
{
	World.Clear();

	auto Light = new DiffuseLight(Vector3(0.8fx, 0.8fx, 0.8fx));
	auto Metal = new MetalMaterial(Vector3(1.fx, 1.fx, 1.fx), 0.4fx);
	auto Glass = new DielectricMaterial(1.5);

	auto Ground = new SphereObject(Vector3(0.fx, -100.5fx, -1.fx), 100.fx, Light);
	auto MetalSphere = new SphereObject(Vector3(0.fx, 0.fx, -1.fx), 0.4fx, Metal);
	auto GlassObject = new SphereObject(Vector3(-1.fx, 0.fx, -1.fx), 0.5fx, Glass);
	World.PushObject(Ground);
	World.PushObject(MetalSphere);
	World.PushObject(GlassObject);
}
void Room3()
{
	World.Clear();

	auto Light = new DiffuseLight(Vector3(1.fx, 1.fx, 1.fx));
	auto Glass = new DielectricMaterial(1.5);

	auto Ground = new SphereObject(Vector3(0.fx, -100.5fx, -1.fx), 100.fx, Light);
	auto GlassObject = new SphereObject(Vector3(0.fx, 0.fx, -1.fx), 0.5fx, Glass);
	World.PushObject(Ground);
	World.PushObject(GlassObject);
}
void Room4()
{
	auto Light = new DiffuseLight(Vector3(1.fx, 1.fx, 1.fx));
	SphereObject *GroudSphere = new SphereObject(Vector3(0.fx, -1000.fx, 0.fx), 1000.fx, Light);

	World.PushObject(GroudSphere);

	int i = 1;
	for (int CountX = -11; CountX < 11; CountX++)
	{
		for (int CountY = -11; CountY < 11; CountY++)
		{
			auto ChooseMat = Random();

			Vector3 Center(static_cast<float>(CountX + 0.9 * Random()), 0.2fx,
						   static_cast<float>(CountY + 0.9 * Random()));
			if ((Center - Vector3(4.fx, 0.2fx, 0.fx)).Length() > 0.9)
			{
				if (ChooseMat < 0.8)
				{
					auto Albedo = Vector3::RandomInUnitSphere() * Vector3::RandomInUnitSphere();
					LambertianMaterial *Material = new LambertianMaterial(new ConstantTexture(Albedo));
					SphereObject *Sphere = new SphereObject(Center, 0.2fx, Material);

					World.PushObject(Sphere);
				}
				else if (ChooseMat < 0.95)
				{
					auto Albedo = Ray::RandomRay(.5fx, 1.fx);
					auto Fuzz = Random(0.fx, .5fx);

					LambertianMaterial *Material = new LambertianMaterial(new ConstantTexture(Albedo));
					SphereObject *Sphere = new SphereObject(Center, 0.2fx, Material);

					World.PushObject(Sphere);
				}
				else
				{
					DielectricMaterial *Material = new DielectricMaterial(1.5fx);
					SphereObject *Sphere = new SphereObject(Center, 0.2fx, Material);

					World.PushObject(Sphere);
				}
			}
		}
	}

	DielectricMaterial *GlassMaterial = new DielectricMaterial(1.5fx);
	LambertianMaterial *Lambertian = new LambertianMaterial(new ConstantTexture(Vector3(0.4fx, 0.2fx, 0.1fx)));
	MetalMaterial *Metal = new MetalMaterial(Vector3(0.7fx, 0.6fx, 0.5fx), 0.fx);

	SphereObject *Sphere1 = new SphereObject(Vector3(0.fx, 1.fx, 0.fx), 1.fx, GlassMaterial);
	SphereObject *Sphere2 = new SphereObject(Vector3(-4.fx, 1.fx, 0.fx), 1.fx, Lambertian);
	SphereObject *Sphere3 = new SphereObject(Vector3(4.fx, 1.fx, 0.fx), 1.fx, Metal);

	World.PushObject(Sphere1);
	World.PushObject(Sphere2);
	World.PushObject(Sphere3);
}
void Room5()
{
	auto Light = new DiffuseLight(Vector3(1.fx, 1.fx, 1.fx));

	LambertianMaterial *Lambertian = new LambertianMaterial(new CheckerTexture(
		new ConstantTexture(Vector3(1.fx, 1.fx, 1.fx)), new ConstantTexture(Vector3(0.fx, 0.fx, 0.fx))));
	SphereObject *GroudSphere = new SphereObject(Vector3(0.fx, -1000.fx, 0.fx), 1000.fx, Lambertian);

	World.PushObject(GroudSphere);

	int i = 1;
	for (int CountX = -11; CountX < 11; CountX++)
	{
		for (int CountY = -11; CountY < 11; CountY++)
		{
			auto ChooseMat = Random();

			Vector3 Center(static_cast<float>(CountX + 0.9 * Random()), 0.2fx,
						   static_cast<float>(CountY + 0.9 * Random()));
			if ((Center - Vector3(4.fx, 0.2fx, 0.fx)).Length() > 0.9)
			{
				if (ChooseMat < 0.8)
				{
					auto Albedo = Vector3::RandomInUnitSphere() * Vector3::RandomInUnitSphere();
					LambertianMaterial *Material = new LambertianMaterial(new ConstantTexture(Albedo));
					SphereObject *Sphere = new SphereObject(Center, 0.2fx, Material);

					World.PushObject(Sphere);
				}
				else if (ChooseMat < 0.95)
				{
					auto Albedo = Ray::RandomRay(.5fx, 1.fx);
					auto Fuzz = Random(0.fx, .5fx);

					LambertianMaterial *Material = new LambertianMaterial(new ConstantTexture(Albedo));
					SphereObject *Sphere = new SphereObject(Center, 0.2fx, Material);

					World.PushObject(Sphere);
				}
				else
				{
					SphereObject *Sphere = new SphereObject(Center, 0.2fx, Light);

					World.PushObject(Sphere);
				}
			}
		}
	}

	DielectricMaterial *GlassMaterial = new DielectricMaterial(1.5fx);
	MetalMaterial *Metal = new MetalMaterial(Vector3(0.7fx, 0.6fx, 0.5fx), 0.fx);

	SphereObject *Sphere1 = new SphereObject(Vector3(0.fx, 1.fx, 0.fx), 1.fx, GlassMaterial);
	SphereObject *Sphere2 = new SphereObject(Vector3(-4.fx, 1.fx, 0.fx), 1.fx, Lambertian);
	SphereObject *Sphere3 = new SphereObject(Vector3(4.fx, 1.fx, 0.fx), 1.fx, Metal);

	World.PushObject(Sphere1);
	World.PushObject(Sphere2);
	World.PushObject(Sphere3);
}

int main()
{
	const int Width = 640;
	const int Height = 480;
	int SamplePerPixel = 200;
	const int MaxRenderDepth = 50;
	bool UseMulThreadRender = true;
	bool UseBVH = false;
	const FloatX AspectRatio = FloatX(Width) / FloatX(Height);

	initgraph(Width, Height, EW_SHOWCONSOLE);

	Vector3 VUP(0.fx, 1.fx, 0.fx);
	WorldCamera = new Camera(Vector3(-2.fx, 2.fx, 1.fx), Vector3(0.fx, 0.fx, -1.fx), VUP, 90.fx, AspectRatio);

	auto Buffer = GetImageBuffer();

	while (true)
	{
		wchar_t Buffer[2];
		InputBox(Buffer, 2, L"将要渲染的场景（1~5）");

		if (Buffer[0] == L'1')
		{
			Room1();
			break;
		}
		if (Buffer[0] == L'2')
		{
			Room2();
			break;
		}
		if (Buffer[0] == L'3')
		{
			Room3();
			break;
		}
		if (Buffer[0] == L'4')
		{
			Room4();
			break;
		}
		if (Buffer[0] == L'5')
		{
			Room5();
			break;
		}
	}

	wchar_t SamplePerPixelString[1024];
	InputBox(SamplePerPixelString, 1024, L"采样率（建议在 200）");
	SamplePerPixel = _wtoi(SamplePerPixelString);

	if (MessageBox(GetHWnd(), L"启用多线程渲染", L"渲染设置", MB_OKCANCEL))
	{
		UseMulThreadRender = true;
	}
	else
	{
		UseMulThreadRender = false;
	}
	if (MessageBox(GetHWnd(), L"启用 BVH", L"渲染设置", MB_OKCANCEL))
	{
		UseBVH = true;
	}
	else
	{
		UseBVH = false;
	}

	BVHWorld = new BVHRoot(World, 0.fx, 1.fx);

	if (UseBVH)
	{
		RenderWorld = BVHWorld;
	}
	else
	{
		RenderWorld = &World;
	}

	time_t Start = clock();

	if (UseMulThreadRender)
	{
		StartThreadRender(Width, Height, MaxRenderDepth, SamplePerPixel);
	}
	else
	{
		auto Buffer = GetImageBuffer();

		for (int Y = Height - 1; Y >= 0; --Y)
		{
			for (int X = 0; X < Width; ++X)
			{
				Vector3 Color;
				/*
				 * 多重采样 MSAA 抗锯齿
				 */
				for (int SampleCount = 0; SampleCount < SamplePerPixel; ++SampleCount)
				{
					auto U = (float)(X + Random()) / Width;
					auto V = (float)(Y + Random()) / Height;

					auto RayInstance = WorldCamera->GetRay(U, V);
					Color += RayColor(RayInstance, RenderWorld, MaxRenderDepth);
				}

				Buffer[OppositeCoordinate(X, Y, Width, Height)] = GetRawColor(Color, SamplePerPixel);
			}
		}
	}

	outtextxy(0, 0, (L"渲染用时：" + std::to_wstring((float(clock()) - Start) / 1000) + L"秒").c_str());
	outtextxy(0, 20, (L"采样率：" + std::wstring(SamplePerPixelString) + L"pixel").c_str());

	if (UseMulThreadRender) { outtextxy(0, 40, L"多线程渲染：开"); }
	else 					{ outtextxy(0, 40, L"多线程渲染：关"); }
	if (UseBVH) 			{ outtextxy(0, 60, L"BVH 优化：开");  }
	else 					{ outtextxy(0, 60, L"BVH 优化：关");  }

	_getch();

	return 0;
}