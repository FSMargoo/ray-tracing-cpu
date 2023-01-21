/*
 *	程序名称 ：光线追踪（Ray Tracing）
 *	作　　者 ：Margoo
 *	邮　　箱 ：1683691371@qq.com
 */
#include <algorithm>
#include <conio.h>
#include <functional>
#include <graphics.h>
#include <intrin.h>
#include <iomanip>
#include <memory>
#include <sstream>
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
	operator float() const
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
 * 三维向量类 Vector3（同时该类也充当了三维点和颜色）
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
	static Vector3 RandomToSphere(FloatX Radius, FloatX DistanceSquared)
	{
		auto R1 = FloatX(::Random());
		auto R2 = FloatX(::Random());
		auto Z = 1.fx + R2 * (FloatX::Sqrt(1.fx - Radius * Radius / DistanceSquared) - 1.fx);

		auto Phi = 2.fx * PI * R1;
		auto X = FloatX::Cos(Phi) * FloatX::Sqrt(1.fx - Z * Z);
		auto Y = FloatX::Sin(Phi) * FloatX::Sqrt(1.fx - Z * Z);

		return Vector3(X, Y, Z);
	}
	static Vector3 Random()
	{
		return Vector3(::Random(), ::Random(), ::Random());
	}
	static Vector3 Random(FloatX Min, FloatX Max)
	{
		return Vector3(::Random(Min, Max), ::Random(Min, Max), ::Random(Min, Max));
	}
	Vector3 &operator/=(const FloatX &Value)
	{
		return *this *= 1.fx / Value;
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

	static Vector3 UnitVector(Vector3 Vector)
	{
		return Vector / Vector.Length();
	}

	static Vector3 RandomUnitVector()
	{
		return UnitVector(RandomInUnitSphere());
	}

	static Vector3 RandomInUnitSphere()
	{
		auto A = ::Random(0.fx, 2.fx * PI);
		auto Z = ::Random(-1.fx, 1.fx);
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
	return Vector3::UnitVector(Vector);
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
 * 标准正交基类
 */
class ONB
{
  public:
	ONB()
	{
	}

	Vector3 operator[](const int &Position) const
	{
		return Axis[Position];
	}

	Vector3 GetU() const
	{
		return Axis[0];
	}
	Vector3 GetV() const
	{
		return Axis[1];
	}
	Vector3 GetW() const
	{
		return Axis[2];
	}

	Vector3 Local(FloatX A, FloatX B, FloatX C) const
	{
		return A * GetU() + B * GetV() + C * GetW();
	}
	Vector3 Local(const Vector3 &A) const
	{
		return A.GetX() * GetU() + A.GetY() * GetV() + A.GetZ() * GetW();
	}

	void BuildFromW(const Vector3 &N)
	{
		Axis[2] = UnitVector(N);

		Vector3 A = (fabs(GetW().GetX()) > 0.9) ? Vector3(0.fx, 1.fx, 0.fx) : Vector3(1.fx, 0.fx, 0.fx);

		Axis[1] = UnitVector(Vector3Cross(GetW(), A));
		Axis[0] = Vector3Cross(GetW(), GetV());
	}

  public:
	Vector3 Axis[3];
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

class PDF;

Vector3 RandomCosineDirection()
{
	auto R1 = FloatX(Random());
	auto R2 = FloatX(Random());
	auto Z = FloatX::Sqrt(1.fx - R2);

	auto PHI = 2.fx * PI * R1;
	auto X = FloatX::Cos(PHI) * FloatX::Sqrt(R2);
	auto Y = FloatX::Sin(PHI) * FloatX::Sqrt(R2);

	return Vector3(X, Y, Z);
}

struct ScatterData
{
	Ray SpecularRay;
	bool IsSpecular;
	Vector3 Attenuation;
	PDF *PDF;
};
class PDF
{
  public:
	virtual ~PDF()
	{
	}

	virtual FloatX GetValue(const Vector3 &Direction) const = 0;
	virtual Vector3 Generate() const = 0;
};

class CosinePDF : public PDF
{
  public:
	CosinePDF(const Vector3 &W)
	{
		UVW.BuildFromW(W);
	}

	FloatX GetValue(const Vector3 &Direction) const override
	{
		auto Cosine = Vector3Dot(UnitVector(Direction), UVW.GetW());

		return (Cosine <= 0.fx) ? 0.fx : Cosine / PI;
	}
	Vector3 Generate() const override
	{
		return UVW.Local(RandomCosineDirection());
	}

  public:
	ONB UVW;
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
	virtual Vector3 Emitted(FloatX U, FloatX V, const HitData &Data, const Vector3 &RayData)
	{
		return Vector3(0.fx, 0.fx, 0.fx);
	}

	virtual bool Scatter(const Ray &RayIn, const HitData &HitData, ScatterData &ScatterData) const
	{
		return false;
	}

	virtual double ScatteringPDF(const Ray &RayIn, const HitData &HitData, const Ray &ScatteredRay) const
	{
		return 0;
	}

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
 * 漫反射材质（又名朗伯材质）
 */
class LambertianMaterial : public ObjectMaterial
{
  public:
	LambertianMaterial(Texture *MaterialTexture) : Texture(MaterialTexture)
	{
	}
	bool Scatter(const Ray &RayIn, const HitData &HitData, ScatterData &ScatterData) const override
	{
		ScatterData.IsSpecular = false;
		ScatterData.Attenuation = Texture->GetValue(HitData.U, HitData.V, HitData.RayData);
		ScatterData.PDF = new CosinePDF(HitData.NormalSurface);

		return true;
	}
	double ScatteringPDF(const Ray &RayIn, const HitData &HitData, const Ray &ScatteredRay) const override
	{
		auto Cosine = Vector3Dot(HitData.NormalSurface, UnitVector(ScatteredRay.Direction));
		return Cosine < 0.fx ? 0.fx : Cosine / PI;
	}

  public:
	Texture *Texture;
};

class MetalMaterial : public ObjectMaterial
{
  public:
	MetalMaterial(const Vector3 &Albedo, FloatX Fuzz) : MaterialAlbedo(Albedo), MetalFuzz(Fuzz < 1.fx ? Fuzz : 1.fx)
	{
	}
	bool Scatter(const Ray &RayIn, const HitData &HitData, ScatterData &ScatterData) const override
	{
		Vector3 Reflected = Reflect(UnitVector(RayIn.Direction), HitData.NormalSurface);

		ScatterData.SpecularRay = Ray(HitData.RayData, Reflected + MetalFuzz * Vector3::RandomInUnitSphere());
		ScatterData.Attenuation = MaterialAlbedo;
		ScatterData.IsSpecular = true;
		ScatterData.PDF = 0;

		return true;
	}

  public:
	FloatX MetalFuzz;
	Vector3 MaterialAlbedo;
};

class DielectricMaterial : public ObjectMaterial
{
  public:
	DielectricMaterial(FloatX RefractiveIndex) : MaterialRefractiveIndex(RefractiveIndex)
	{
	}
	bool Scatter(const Ray &RayIn, const HitData &HitData, ScatterData &ScatterData) const override
	{
		ScatterData.IsSpecular = true;
		ScatterData.PDF = nullptr;
		ScatterData.Attenuation = Vector3(1.0fx, 1.0fx, 1.0fx);
		FloatX RefractionRatio = HitData.FrontFace ? (1.0fx / MaterialRefractiveIndex) : MaterialRefractiveIndex;

		Vector3 UnitDirection = UnitVector(RayIn.Direction);
		FloatX CosTheta = FloatX::Min(Vector3Dot(-UnitDirection, HitData.NormalSurface), 1.0fx);
		FloatX SinTheta = FloatX::Sqrt(1.0fx - CosTheta * CosTheta);

		bool AbleToRefract = RefractionRatio * SinTheta > 1.0;
		Vector3 Direction;

		if (AbleToRefract || Schlick(CosTheta, RefractionRatio) > Random())
		{
			Direction = Reflect(UnitDirection, HitData.NormalSurface);
		}
		else
		{
			Direction = Refract(UnitDirection, HitData.NormalSurface, RefractionRatio);
		}

		ScatterData.SpecularRay = Ray(HitData.RayData, Direction);

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

	/*
	 * 光源则返回光源的颜色
	 */
	Vector3 Emitted(FloatX U, FloatX V, const HitData &Data, const Vector3 &RayData) override
	{
		if (Data.FrontFace)
		{
			return LightColor;
		}
		else
		{
			return Vector3(0.fx, 0.fx, 0.fx);
		}
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

	virtual FloatX PDFValue(const Vector3 &OriginPoint, const Vector3 &Vector) const
	{
		return 0.fx;
	}
	virtual Vector3 Random(const Vector3 &Origin) const
	{
		return Vector3(1.fx, 0.fx, 0.fx);
	}

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
class ObjectHitPDF : public PDF
{
  public:
	ObjectHitPDF(HitTestBase *Object, const Vector3 &Origin) : ObjectRef(Object), OriginPoint(Origin)
	{
	}

	FloatX GetValue(const Vector3 &Direction) const override
	{
		return ObjectRef->PDFValue(OriginPoint, Direction);
	}
	Vector3 Generate() const override
	{
		return ObjectRef->Random(OriginPoint);
	}

  public:
	Vector3 OriginPoint;
	HitTestBase *ObjectRef;
};

class XYRect : public HitTestBase
{
  public:
	XYRect() : Material(nullptr)
	{
	}

	XYRect(FloatX IX0, FloatX IX1, FloatX IY0, FloatX IY1, FloatX IK, ObjectMaterial *IMaterial)
		: X0(IX0), X1(IX1), Y0(IY0), Y1(IY1), K(IK), Material(IMaterial)
	{
	}

	FloatX PDFValue(const Vector3 &Origin, const Vector3 &Vector) const override
	{
		HitData Data;
		if (!HitTest(Ray(Origin, Vector), 0.001fx, Inf, Data))
		{
			return 0.fx;
		}

		auto Area = (X1 - X0) * (Y1 - Y0);
		auto DistanceSuqared = Data.T * Data.T * Vector.LengthSquared();
		auto Cosine = FloatX(fabs(Vector3Dot(Vector, Data.NormalSurface) / Vector.Length()));

		return DistanceSuqared / (Cosine * Area);
	}
	Vector3 Random(const Vector3 &Origin) const override
	{
		auto RandomPoint = Vector3(::Random(X0, X1), ::Random(Y0, Y1), K);
		return RandomPoint - Origin;
	}

	bool HitTest(const Ray &RayInstance, FloatX T0, FloatX T1, HitData &Data) const override
	{
		auto T = (K - RayInstance.LightOrigin.GetZ()) / RayInstance.Direction.GetZ();
		if (T < T0 || T > T1)
		{
			return false;
		}

		auto X = RayInstance.LightOrigin.GetX() + T * RayInstance.Direction.GetX();
		auto Y = RayInstance.LightOrigin.GetY() + T * RayInstance.Direction.GetY();

		float OX0;
		float OX1;
		float OY0;
		float OY1;

		OX0 = (const float)X0;
		OX1 = (const float)X1;
		OY0 = (const float)Y0;
		OY1 = (const float)Y1;

		if (X < OX0 || X > OX1 || Y < OY0 || Y > OY1)
		{
			return false;
		}

		Data.U = (X - OX0) / (OX1 - OX0);
		Data.V = (Y - OY0) / (OY1 - OY0);
		Data.T = T;

		Vector3 OutwardNormal = Vector3(0.fx, 0.fx, 1.fx);

		Data.SetSurfaceNormal(RayInstance, OutwardNormal);
		Data.Material = Material;
		Data.RayData = RayInstance.LightAt(T);

		return true;
	}
	bool HittingBox(FloatX T0, FloatX T1, BVHBox &Box) const override
	{
		Box = BVHBox(Vector3(X0, Y0, K - 0.0001fx), Vector3(X1, Y1, K + 0.001fx));

		return true;
	}

  public:
	FloatX X0;
	FloatX X1;
	FloatX Y0;
	FloatX Y1;
	FloatX K;

	ObjectMaterial *Material;
};
class XZRect : public HitTestBase
{
  public:
	XZRect() : Material(nullptr)
	{
	}

	XZRect(FloatX IX0, FloatX IX1, FloatX IZ0, FloatX IZ1, FloatX IK, ObjectMaterial *IMaterial)
		: X0(IX0), X1(IX1), Z0(IZ0), Z1(IZ1), K(IK), Material(IMaterial)
	{
	}

	bool HitTest(const Ray &RayInstance, FloatX T0, FloatX T1, HitData &Data) const override
	{
		auto T = (K - RayInstance.LightOrigin.GetY()) / RayInstance.Direction.GetY();
		if (T < T0 || T > T1)
		{
			return false;
		}

		auto X = RayInstance.LightOrigin.GetX() + T * RayInstance.Direction.GetX();
		auto Z = RayInstance.LightOrigin.GetZ() + T * RayInstance.Direction.GetZ();

		float OX0;
		float OX1;
		float OZ0;
		float OZ1;

		OX0 = (const float)X0;
		OX1 = (const float)X1;
		OZ0 = (const float)Z0;
		OZ1 = (const float)Z1;

		if (X < OX0 || X > OX1 || Z < OZ0 || Z > OZ1)
		{
			return false;
		}

		Data.U = (X - OX0) / (OX1 - OX0);
		Data.V = (Z - OZ0) / (OZ1 - OZ0);
		Data.T = T;

		Vector3 OutwardNormal = Vector3(0.fx, 1.fx, 0.fx);

		Data.SetSurfaceNormal(RayInstance, OutwardNormal);
		Data.Material = Material;
		Data.RayData = RayInstance.LightAt(T);

		return true;
	}
	bool HittingBox(FloatX T0, FloatX T1, BVHBox &Box) const override
	{
		Box = BVHBox(Vector3(X0, K - 0.0001fx, Z0), Vector3(X1, K + 0.001fx, Z1));

		return true;
	}

	FloatX PDFValue(const Vector3 &Origin, const Vector3 &Vector) const override
	{
		HitData Data;
		if (!HitTest(Ray(Origin, Vector), 0.001fx, Inf, Data))
		{
			return 0.fx;
		}

		auto Area = (X1 - X0) * (Z1 - Z0);
		auto DistanceSuqared = Data.T * Data.T * Vector.LengthSquared();
		auto Cosine = FloatX(fabs(Vector3Dot(Vector, Data.NormalSurface) / Vector.Length()));

		return DistanceSuqared / (Cosine * Area);
	}
	Vector3 Random(const Vector3 &Origin) const override
	{
		auto RandomPoint = Vector3(::Random(X0, X1), K, ::Random(Z0, Z1));
		return RandomPoint - Origin;
	}

  public:
	FloatX X0;
	FloatX X1;
	FloatX Z0;
	FloatX Z1;
	FloatX K;

	ObjectMaterial *Material;
};
class YZRect : public HitTestBase
{
  public:
	YZRect() : Material(nullptr)
	{
	}

	YZRect(FloatX IY0, FloatX IY1, FloatX IZ0, FloatX IZ1, FloatX IK, ObjectMaterial *IMaterial)
		: Y0(IY0), Y1(IY1), Z0(IZ0), Z1(IZ1), K(IK), Material(IMaterial)
	{
	}

	FloatX PDFValue(const Vector3 &Origin, const Vector3 &Vector) const override
	{
		HitData Data;
		if (!HitTest(Ray(Origin, Vector), 0.001fx, Inf, Data))
		{
			return 0.fx;
		}

		auto Area = (Y1 - Y0) * (Y1 - Y0);
		auto DistanceSuqared = Data.T * Data.T * Vector.LengthSquared();
		auto Cosine = FloatX(fabs(Vector3Dot(Vector, Data.NormalSurface) / Vector.Length()));

		return DistanceSuqared / (Cosine * Area);
	}
	Vector3 Random(const Vector3 &Origin) const override
	{
		auto RandomPoint = Vector3(K, ::Random(Y0, Y1), ::Random(Y0, Y1));
		return RandomPoint - Origin;
	}

	bool HitTest(const Ray &RayInstance, FloatX T0, FloatX T1, HitData &Data) const override
	{
		auto T = (K - RayInstance.LightOrigin.GetX()) / RayInstance.Direction.GetX();
		if (T < T0 || T > T1)
		{
			return false;
		}

		auto Y = RayInstance.LightOrigin.GetY() + T * RayInstance.Direction.GetY();
		auto Z = RayInstance.LightOrigin.GetZ() + T * RayInstance.Direction.GetZ();

		float OY0;
		float OY1;
		float OZ0;
		float OZ1;

		OY0 = (const float)Y0;
		OY1 = (const float)Y1;
		OZ0 = (const float)Z0;
		OZ1 = (const float)Z1;

		if (Y < OY0 || Y > OY1 || Z < OZ0 || Z > OZ1)
		{
			return false;
		}

		Data.U = (Y - OY0) / (OY1 - OY0);
		Data.V = (Z - OZ0) / (OZ1 - OZ0);
		Data.T = T;

		Vector3 OutwardNormal = Vector3(1.fx, 0.fx, 0.fx);

		Data.SetSurfaceNormal(RayInstance, OutwardNormal);
		Data.Material = Material;
		Data.RayData = RayInstance.LightAt(T);

		return true;
	}
	bool HittingBox(FloatX T0, FloatX T1, BVHBox &BoY) const override
	{
		BoY = BVHBox(Vector3(K - 0.0001fx, Y0, Z0), Vector3(K + 0.0001fx, Y1, Z1));

		return true;
	}

  public:
	FloatX Y0;
	FloatX Y1;
	FloatX Z0;
	FloatX Z1;
	FloatX K;

	ObjectMaterial *Material;
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

	FloatX PDFValue(const Vector3 &Origin, const Vector3 &Vector) const override
	{
		HitData Data;
		if (!HitTest(Ray(Origin, Vector), 0.001fx, Inf, Data))
		{
			return 0.fx;
		}

		auto CosThetaMax = FloatX::Sqrt(1.fx - Radius * Radius / (CenterPoint - Origin).LengthSquared());
		auto SolidAngle = 2.fx * PI * (1.fx - CosThetaMax);

		return 1.fx / SolidAngle;
	}
	Vector3 Random(const Vector3 &Origin) const override
	{
		Vector3 Direction = CenterPoint - Origin;
		auto DistanceSquared = Direction.LengthSquared();

		ONB UVW;
		UVW.BuildFromW(Direction);

		return UVW.Local(Vector3::RandomToSphere(Radius, DistanceSquared));
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
class FlipFace : public HitTestBase
{
  public:
	FlipFace(HitTestBase *Object) : ObjectRef(Object)
	{
	}

	bool HitTest(const Ray &Instance, FloatX TMin, FloatX TMax, HitData &Data) const override
	{
		if (!ObjectRef->HitTest(Instance, TMin, TMax, Data))
		{
			return false;
		}

		Data.FrontFace = !Data.FrontFace;
		return true;
	}
	bool HittingBox(FloatX T0, FloatX T1, BVHBox &Box) const override
	{
		return ObjectRef->HittingBox(T0, T1, Box);
	}

  public:
	HitTestBase *ObjectRef;
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

	FloatX PDFValue(const Vector3 &Origin, const Vector3 &Vector) const override
	{
		auto Weight = 1.fx / FloatX(Objects.size());
		auto Sum = 0.fx;

		for (auto Object : Objects)
		{
			Sum += Weight * Object->PDFValue(Origin, Vector);
		}

		return Sum;
	}
	Vector3 Random(const Vector3 &OriginPoint) const override
	{
		if (Objects.size() > 1)
		{
			return Objects[rand() % (Objects.size())]->Random(OriginPoint);
		}
		else if (!Objects.empty())
		{
			return Objects[0]->Random(OriginPoint);
		}
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

class Translate : public HitTestBase
{
  public:
	Translate(HitTestBase *Object, const Vector3 &Displacement) : ObjectRef(Object), PlaceOffset(Displacement)
	{
	}

	bool HitTest(const Ray &RayInstance, FloatX TMin, FloatX TMax, HitData &Data) const override
	{
		Ray OffsetRay(RayInstance.LightOrigin - PlaceOffset, RayInstance.Direction);
		if (!ObjectRef->HitTest(OffsetRay, TMin, TMax, Data))
		{
			return false;
		}

		Data.RayData += PlaceOffset;
		Data.SetSurfaceNormal(OffsetRay, Data.NormalSurface);

		return true;
	}
	bool HittingBox(FloatX T0, FloatX T1, BVHBox &Box) const override
	{
		if (!ObjectRef->HittingBox(T0, T1, Box))
		{
			return false;
		}

		Box = BVHBox(Box.BoxMin + PlaceOffset, Box.BoxMax + PlaceOffset);

		return true;
	}

  public:
	HitTestBase *ObjectRef;
	Vector3 PlaceOffset;
};
class RotateY : public HitTestBase
{
  public:
	RotateY(HitTestBase *Object, FloatX RotateAngle)
	{
		ObjectRef = Object;
		auto Radians = DegreesToRadians(RotateAngle);
		SinTheta = FloatX::Sin(Radians);
		CosTheta = FloatX::Cos(Radians);
		HasBox = Object->HittingBox(0.fx, 1.fx, Box);

		Vector3 Min(Inf, Inf, Inf);
		Vector3 Max(-Inf, -Inf, -Inf);

		for (FloatX I = 0.fx; I < 2.fx; I += 1.fx)
		{
			for (FloatX J = 0.fx; J < 2.fx; J += 1.fx)
			{
				for (FloatX K = 0.fx; K < 2.fx; K += 1.fx)
				{
					auto X = I * Box.BoxMax.GetX() + (1.fx - I) * Box.BoxMin.GetX();
					auto Y = J * Box.BoxMax.GetY() + (1.fx - J) * Box.BoxMin.GetY();
					auto Z = K * Box.BoxMax.GetZ() + (1.fx - K) * Box.BoxMin.GetZ();

					auto NewX = CosTheta * X + SinTheta * Z;
					auto NewZ = -SinTheta * X + CosTheta * Z;

					Vector3 Resever(NewX, Y, NewZ);

					for (FloatX M = 0.fx; M < 3.fx; M += 1.fx)
					{
						Min[M] = FloatX::Min(Min[M], Resever[M]);
						Max[M] = FloatX::Max(Max[M], Resever[M]);
					}
				}
			}
		}

		Box = BVHBox(Min, Max);
	}
	bool HitTest(const Ray &RayInstance, FloatX TMin, FloatX TMax, HitData &Data) const
	{
		Vector3 LightOrigin = RayInstance.LightOrigin;
		Vector3 LightDirection = RayInstance.Direction;

		LightOrigin[0] = CosTheta * RayInstance.LightOrigin[0] - SinTheta * RayInstance.LightOrigin[2];
		LightOrigin[2] = SinTheta * RayInstance.LightOrigin[0] + CosTheta * RayInstance.LightOrigin[2];

		LightDirection[0] = CosTheta * RayInstance.Direction[0] - SinTheta * RayInstance.Direction[2];
		LightDirection[2] = SinTheta * RayInstance.Direction[0] + CosTheta * RayInstance.Direction[2];

		Ray RotatedRay(LightOrigin, LightDirection);
		if (!ObjectRef->HitTest(RotatedRay, TMin, TMax, Data))
		{
			return false;
		}

		Vector3 RayData = Data.RayData;
		Vector3 SurfaceNormal = Data.NormalSurface;

		RayData[0] = CosTheta * Data.RayData[0] + SinTheta * Data.RayData[2];
		RayData[2] = -SinTheta * Data.RayData[0] + CosTheta * Data.RayData[2];

		SurfaceNormal[0] = CosTheta * Data.NormalSurface[0] + SinTheta * Data.NormalSurface[2];
		SurfaceNormal[2] = -SinTheta * Data.NormalSurface[0] + CosTheta * Data.NormalSurface[2];

		Data.RayData = RayData;
		Data.SetSurfaceNormal(RotatedRay, SurfaceNormal);

		return true;
	}
	bool HittingBox(FloatX T0, FloatX T1, BVHBox &OutputBox) const
	{
		OutputBox = Box;

		return HasBox;
	}

  public:
	HitTestBase *ObjectRef;
	FloatX SinTheta;
	FloatX CosTheta;
	bool HasBox;
	BVHBox Box;
};

class Cube : public HitTestBase
{
  public:
	Cube() = default;
	Cube(const Vector3 &Point0, const Vector3 &Point1, ObjectMaterial *Material)
	{
		BoxMin = Point0;
		BoxMax = Point1;

		Sides.PushObject(
			new XYRect(Point0.GetX(), Point1.GetX(), Point0.GetY(), Point1.GetY(), Point1.GetZ(), Material));
		Sides.PushObject(new FlipFace(
			new XYRect(Point0.GetX(), Point1.GetX(), Point0.GetY(), Point1.GetY(), Point0.GetZ(), Material)));

		Sides.PushObject(
			new XZRect(Point0.GetX(), Point1.GetX(), Point0.GetZ(), Point1.GetZ(), Point1.GetY(), Material));
		Sides.PushObject(new FlipFace(
			new XZRect(Point0.GetX(), Point1.GetX(), Point0.GetZ(), Point1.GetZ(), Point0.GetY(), Material)));

		Sides.PushObject(
			new YZRect(Point0.GetY(), Point1.GetY(), Point0.GetZ(), Point1.GetZ(), Point1.GetX(), Material));
		Sides.PushObject(new FlipFace(
			new YZRect(Point0.GetY(), Point1.GetY(), Point0.GetZ(), Point1.GetZ(), Point0.GetX(), Material)));
	}
	bool HitTest(const Ray &RayInstance, FloatX T0, FloatX T1, HitData &Data) const override
	{
		return Sides.HitTest(RayInstance, T0, T1, Data);
	}
	bool HittingBox(FloatX T0, FloatX T1, BVHBox &Box) const
	{
		Box = BVHBox(BoxMin, BoxMax);

		return true;
	}

  public:
	Vector3 BoxMin;
	Vector3 BoxMax;
	TraceWorld Sides;
};

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
	static Vector3 RandomInUnitDisk()
	{
		while (true)
		{
			auto Point = Vector3(Random(-1.fx, 1.fx), Random(-1.fx, 1.fx), 0.fx);
			if (Point.LengthSquared() >= 1.fx)
			{
				continue;
			}

			return Point;
		}
	}

  public:
	Camera(Vector3 LookFrom, Vector3 LookAt, Vector3 VUP, FloatX VFov, FloatX Aspect, FloatX Aperture, FloatX FocusDist)
	{
		OriginPoint = LookFrom;
		LensRadius = Aperture / 2.fx;

		auto Theta = DegreesToRadians(VFov);
		auto ViewHeight = 2.0fx * FloatX::Tan(Theta / 2.fx);
		auto ViewWidth = Aspect * ViewHeight;

		W = UnitVector(LookFrom - LookAt);
		U = UnitVector(Vector3Cross(VUP, W));
		V = Vector3Cross(W, U);

		Horizontal = FocusDist * ViewWidth * U;
		Vertical = FocusDist * ViewHeight * V;

		LowerLeftCorner = OriginPoint - Horizontal / 2 - Vertical / 2 - FocusDist * W;
	}
	Camera(Vector3 LookFrom, Vector3 LookAt, Vector3 VUP, FloatX VFov, FloatX Aspect)
		: Camera(LookFrom, LookAt, VUP, VFov, Aspect, 0.fx, 1.fx)
	{
	}

	Ray GetRay(FloatX S, FloatX T)
	{
		Vector3 Radius = LensRadius * RandomInUnitDisk();
		Vector3 Offset = U * Radius.GetX() + V * Radius.GetY();

		return Ray(OriginPoint + Offset, LowerLeftCorner + S * Horizontal + T * Vertical - OriginPoint - Offset);
	}

  public:
	Vector3 OriginPoint;
	Vector3 LowerLeftCorner;
	Vector3 Horizontal;
	Vector3 Vertical;
	Vector3 U;
	Vector3 V;
	Vector3 W;
	FloatX LensRadius;
};

class MixturePDF : public PDF
{
  public:
	MixturePDF(PDF *Object1, PDF *Object2)
	{
		PDFList[0] = Object1;
		PDFList[1] = Object2;
	}

	FloatX GetValue(const Vector3 &Direction) const override
	{
		return 0.5fx * PDFList[0]->GetValue(Direction) + 0.5fx * PDFList[1]->GetValue(Direction);
	}

	Vector3 Generate() const override
	{
		if (Random() < 0.5fx)
		{
			return PDFList[0]->Generate();
		}
		else
		{
			return PDFList[1]->Generate();
		}
	}

  public:
	PDF *PDFList[2];
};

bool IsBadPoint(const Vector3 &Vector)
{
	auto R = float(Vector[0]);
	auto G = float(Vector[1]);
	auto B = float(Vector[2]);

	bool Flag = false;

	if (R != R || G != G || B != B)
	{
		Flag = true;
	}

	return Flag;
}

DWORD GetRawColor(const Vector3 &Vector, FloatX SamplePerPixel)
{
	auto R = Vector[0];
	auto G = Vector[1];
	auto B = Vector[2];

	if (R != R)
	{
		R = 0.f;
	}
	if (G != G)
	{
		G = 0.f;
	}
	if (B != B)
	{
		B = 0.f;
	}

	auto Scale = 1.fx / SamplePerPixel;

	R = FloatX::Sqrt(Scale * R);
	G = FloatX::Sqrt(Scale * G);
	B = FloatX::Sqrt(Scale * B);

	return RGB(Range(B, 0.fx, 0.999fx) * 256.fx, Range(G, 0.fx, 0.999fx) * 256.fx, Range(R, 0.fx, 0.999fx) * 256.fx);
}

/*
 * 实际上的渲染是由下到上地渲染，需要镜面反转渲染的点
 */
template <class Type> Type OppositeCoordinate(const Type &X, const Type &Y, const Type &Width, const Type &Height)
{
	return (Height - Y - 1) * Width + X;
}

TraceWorld GlobalLight;
/*
 * 渲染函数
 */
Vector3 RayColor(const Ray &RayObject, HitTestBase *World, HitTestBase *Light, int Depth)
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
	Vector3 Emitted = Data.Material->Emitted(Data.U, Data.V, Data, Data.RayData);
	FloatX PDFValue;

	ScatterData ScatterData;

	if (!Data.Material->Scatter(RayObject, Data, ScatterData))
	{
		return Emitted;
	}

	if (ScatterData.IsSpecular)
	{
		return ScatterData.Attenuation * RayColor(ScatterData.SpecularRay, World, Light, Depth - 1);
	}

	auto LightPDF = new ObjectHitPDF(&GlobalLight, Data.RayData);
	MixturePDF MixedPDF(LightPDF, ScatterData.PDF);

	ScatteredRay = Ray(Data.RayData, MixedPDF.Generate());
	PDFValue = MixedPDF.GetValue(ScatteredRay.Direction);

	delete LightPDF;
	delete ScatterData.PDF;

	return Emitted + ScatterData.Attenuation * Data.Material->ScatteringPDF(RayObject, Data, ScatteredRay) *
						 RayColor(ScatteredRay, World, Light, Depth - 1) / PDFValue;
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

			while (true)
			{
				Color = Vector3(0.fx, 0.fx, 0.fx);

				for (int SampleCount = 0; SampleCount < SamplePerPixel; ++SampleCount)
				{
					auto U = (float)(X + Random()) / Width;
					auto V = (float)(Y + Random()) / Height;

					auto RayInstance = WorldCamera->GetRay(U, V);
					Color += RayColor(RayInstance, RenderWorld, &GlobalLight, MaxRenderDepth);
				}

				if (!IsBadPoint(Color))
				{
					break;
				}
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
	Vector3 VUP(0.fx, 1.fx, 0.fx);
	WorldCamera = new Camera(Vector3(-2.fx, 2.fx, 1.fx), Vector3(0.fx, 0.fx, -1.fx), VUP, 90.fx, 640.f / 480.f);

	auto Light = new DiffuseLight(Vector3(1.fx, 1.fx, 1.fx));
	SphereObject *GroudSphere = new SphereObject(Vector3(0.fx, -1000.fx, 0.fx), 1000.fx, Light);

	GlobalLight.PushObject(GroudSphere);
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
void Room2()
{
	Vector3 VUP(0.fx, 1.fx, 0.fx);
	WorldCamera = new Camera(Vector3(-2.fx, 2.fx, 1.fx), Vector3(0.fx, 0.fx, -1.fx), VUP, 90.fx, 640.f / 480.f);

	auto Light = new DiffuseLight(Vector3(0.7fx, 0.7fx, 0.7fx));

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

					if (GlobalLight.Objects.empty())
					{
						GlobalLight.PushObject(Sphere);
					}
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
void Room3()
{
	Vector3 VUP(0.fx, 1.fx, 0.fx);
	WorldCamera =
		new Camera(Vector3(278.fx, 278.fx, -800.fx), Vector3(278.fx, 278.fx, 0.fx), VUP, 40.fx, 640.f / 480.f);

	World.Clear();

	auto Red = new LambertianMaterial(new ConstantTexture(Vector3(0.65, 0.05, 0.05)));
	auto White = new LambertianMaterial(new ConstantTexture(Vector3(0.73, 0.73, 0.73)));
	auto Metal = new MetalMaterial(Vector3(0.8fx, 0.85fx, 0.88fx), 0.0fx);
	auto Green = new LambertianMaterial(new ConstantTexture(Vector3(0.12, 0.45, 0.15)));
	auto Light = new DiffuseLight(Vector3(15.fx, 15.fx, 15.fx));

	auto LightObject = new FlipFace(new XZRect(213, 343, 227, 332, 554, Light));

	World.PushObject(new FlipFace(new YZRect(0, 555, 0, 555, 555, Green)));
	World.PushObject(new YZRect(0, 555, 0, 555, 0, Red));
	World.PushObject(LightObject);
	World.PushObject(new FlipFace(new XZRect(0, 555, 0, 555, 555, White)));
	World.PushObject(new XZRect(0, 555, 0, 555, 0, White));
	World.PushObject(new FlipFace(new XYRect(0, 555, 0, 555, 555, White)));

	GlobalLight.PushObject(LightObject->ObjectRef);

	World.PushObject(
		new Translate(new RotateY(new Cube(Vector3(0.fx, 0.fx, 0.fx), Vector3(165.fx, 330.fx, 165.fx), White), 15.fx),
					  Vector3(265.fx, 0.fx, 295.fx)));
	World.PushObject(new Translate(new RotateY(new SphereObject(Vector3(190.fx, 90.fx, 190.fx), 90.fx, Metal), -18.fx),
								   Vector3(130.fx, 0.fx, 65.fx)));
}
void Room4()
{
	Vector3 VUP(0.fx, 1.fx, 0.fx);
	WorldCamera = new Camera(Vector3(478.fx, 278.fx, -600.fx), Vector3(278.fx, 278.fx, 0.fx), VUP, 40.fx,
							 640.fx / 480.fx, 0.fx, 10.fx);

	auto Ground = new LambertianMaterial(new ConstantTexture(Vector3(0.48, 0.83, 0.53)));

	const int BoxesCount = 20.fx;
	for (int I = 0; I < BoxesCount; I++)
	{
		for (int J = 0; J < BoxesCount; J++)
		{
			auto W = 100.fx;
			auto X0 = -1000.fx + FloatX(I) * W;
			auto Z0 = -1000.fx + FloatX(J) * W;
			auto Y0 = 0.fx;
			auto X1 = X0 + W;
			auto Y1 = Random(1.fx, 101.fx);
			auto Z1 = Z0 + W;

			World.PushObject(new Cube(Vector3(X0, Y0, Z0), Vector3(X1, Y1, Z1), Ground));
		}
	}

	World.PushObject(new SphereObject(Vector3(360.fx, 150.fx, 145.fx), 70.fx, new DielectricMaterial(1.5fx)));

	auto Light = new DiffuseLight(Vector3(10.fx, 10.fx, 10.fx));
	auto LightBox = new FlipFace(new XZRect(123, 423, 147, 412, 550.fx, Light));

	TraceWorld *Boxes = new TraceWorld;
	auto White = new LambertianMaterial(new ConstantTexture(Vector3(0.73, 0.73, 0.73)));
	int SphereCount = 1000;
	for (int Count = 0; Count < SphereCount; ++Count)
	{
		Boxes->PushObject(new SphereObject(Vector3::Random(0.fx, 165.fx), 10, White));
	}

	World.PushObject(new SphereObject(Vector3(400.fx, 200.fx, 400.fx), 100.fx,
									  new MetalMaterial(Vector3(0.7fx, 0.68fx, 0.6fx), 0.fx)));
	World.PushObject(new Translate(new RotateY(new BVHNode(*Boxes, 0.0fx, 1.0fx), 15), Vector3(-100, 270, 395)));

	World.PushObject(LightBox);
	GlobalLight.PushObject(LightBox->ObjectRef);
}

int main()
{
	const int Width = 640;
	const int Height = 480;
	int SamplePerPixel = 100;
	const int MaxRenderDepth = 50;
	bool UseMulThreadRender = true;
	bool UseBVH = true;
	const FloatX AspectRatio = FloatX(Width) / FloatX(Height);

	initgraph(Width, Height, EW_SHOWCONSOLE);

	auto Buffer = GetImageBuffer();
	int RoomNumber = 0;

	while (true)
	{
		wchar_t Buffer[2];
		InputBox(Buffer, 2, L"将要渲染的场景（1~4）");

		if (Buffer[0] == L'1')
		{
			RoomNumber = 1;

			Room1();
			break;
		}
		if (Buffer[0] == L'2')
		{
			RoomNumber = 2;

			Room2();
			break;
		}
		if (Buffer[0] == L'3')
		{
			RoomNumber = 3;

			Room3();
			break;
		}
		if (Buffer[0] == L'4')
		{
			RoomNumber = 4;

			Room4();
			break;
		}
	}

	wchar_t SamplePerPixelString[1024];
	InputBox(SamplePerPixelString, 1024, L"采样率（1～10000），默认 200。采样率越高，效果越好，速度越慢。");
	SamplePerPixel = _wtoi(SamplePerPixelString);

	if (MessageBox(GetHWnd(), L"启用多线程渲染", L"渲染设置", MB_OKCANCEL) != IDCANCEL)
	{
		UseMulThreadRender = true;
	}
	else
	{
		UseMulThreadRender = false;
	}
	if (MessageBox(GetHWnd(), L"启用 BVH", L"渲染设置", MB_OKCANCEL) != IDCANCEL)
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
					Color += RayColor(RayInstance, RenderWorld, &GlobalLight, MaxRenderDepth);
				}

				Buffer[OppositeCoordinate(X, Y, Width, Height)] = GetRawColor(Color, SamplePerPixel);
			}
		}
	}

	std::wstringstream StringStream;
	StringStream << std::setprecision(7) << float(clock() - Start) / 1000.f;

	outtextxy(0, 0, (L"渲染用时：" + StringStream.str() + L" 秒").c_str());
	outtextxy(0, 20, (L"采样率：" + std::to_wstring(SamplePerPixel) + L" spp").c_str());

	if (UseMulThreadRender)
	{
		outtextxy(0, 40, L"多线程渲染：开");
	}
	else
	{
		outtextxy(0, 40, L"多线程渲染：关");
	}
	if (UseBVH)
	{
		outtextxy(0, 60, L"BVH 优化：开");
	}
	else
	{
		outtextxy(0, 60, L"BVH 优化：关");
	}

	outtextxy(0, 80, (L"Room " + std::to_wstring(RoomNumber)).c_str());

	_getch();

	return 0;
}