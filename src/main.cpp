#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <vector>
#include <math.h>
#include <limits>

using std::cos;

// temporary declarations
namespace jp
{
	typedef std::pair<cv::Mat, cv::Mat> cv_trans_t;
}

inline bool containsNaNs(const cv::Mat& m)
{
	return cv::sum(cv::Mat(m != m))[0] > 0;
}

void print(std::string name, cv::Mat& mat)
{
	std::cout << "matrix: " << name << std::endl;
	std::cout << "rows: " << mat.rows << std::endl;
	std::cout << "cols: " << mat.cols << std::endl;
	std::cout << "channels: " << mat.channels() << std::endl;
	std::cout << mat << std::endl << std::endl;
}
////////

class Transformation
{
public:
	Transformation(cv::Vec3f* rots=nullptr, cv::Vec3f* trans=nullptr)
	{
//		r = cv::Mat_<double>::zeros(1, 3);
		t = cv::Mat_<double>::zeros(1, 3);

		if (rots)
			this->rots = *rots;
		else
			cv::randu(this->rots, cv::Scalar(0), cv::Scalar(2*M_PI));

		if (trans)
		{
			t.at<double>(0,0) = (*trans)[0];
			t.at<double>(0,1) = (*trans)[1];
			t.at<double>(0,2) = (*trans)[2];
		}
		else
			cv::randn(t, 0, 1);

		cv::Rodrigues(computeRotationMatrix(), r);
//		std::cout << computeRotationMatrix() << std::endl;

	}

	cv::Mat computeRotationMatrix()
	{
//		std::cout << rots << std::endl;
		cv::Mat rotX = (cv::Mat_<float> (3, 3) <<
						1, 0, 0,
						0, cos(rots[0]), -sin(rots[0]),
						0, sin(rots[0]), cos(rots[0])
						);
		cv::Mat rotY = (cv::Mat_<float> (3, 3) <<
						cos(rots[1]), 0, -sin(rots[1]),
						0, 1, 0,
						sin(rots[1]), 0, cos(rots[1])
						);
		cv::Mat rotZ = (cv::Mat_<float> (3, 3) <<
						cos(rots[2]), -sin(rots[2]), 0,
						sin(rots[2]), cos(rots[2]), 0,
						0, 0, 1
						);

		return rotZ * (rotY * rotX);
	}

	cv::Vec3f getRotationAngles()
	{
		return rots;
	}

	cv::Mat getRotationVector()
	{
		return r;
	}

	cv::Mat getRotationMatrix()
	{
		cv::Mat R;
		cv::Rodrigues(r, R);
		return R;
	}

	cv::Mat getTranslation()
	{
		return t;
	}

	void setRotationAngles(const cv::Vec3f& rots)
	{
		this->rots = rots;
		cv::Rodrigues(computeRotationMatrix(), r);
	}

	void setTranslation(const cv::Mat& trans)
	{
		t = trans;
	}

private:
	cv::Vec3f rots;
	cv::Mat r;
	cv::Mat t;
};

/*
 * Helper functions
 */
std::vector<cv::Point3f> createScene(unsigned int N, int* seed = nullptr)
{
	std::cout << "create scene " << std::endl;

	std::vector<cv::Point3f> data;

	if (!seed)
	{
		data.push_back(cv::Point3f(.15, .2, .3));
		data.push_back(cv::Point3f(.4, .5, .6));
		data.push_back(cv::Point3f(.7, .8, .9));
		data.push_back(cv::Point3f(.2, .1, .2));

		return data;
	}
	else
	{

	}

	return data;
}

std::vector<cv::Point3f> createMeasurements(std::vector<cv::Point3f>& data, Transformation& transform, bool fixed=false)
{
	std::cout << "create measurements " << std::endl;

	std::vector<cv::Point3f> measurements;

	if (!fixed)
	{
		for (unsigned int i = 0; i < data.size(); ++i)
		{
			cv::Mat_<double> res = transform.getRotationMatrix() * cv::Mat(data[i], false);
			res += transform.getTranslation().t();
			measurements.push_back(cv::Point3f(res.at<double>(0,0), res.at<double>(1,0), res.at<double>(2,0)));
		}
	}

	return measurements;
}

cv::Mat svd_backward(const std::vector<cv::Mat> &grads, const cv::Mat& self, bool some, const cv::Mat& raw_u, const cv::Mat& sigma, const cv::Mat& raw_v)
{
	auto m = self.rows;
	auto n = self.cols;
	auto k = sigma.cols;
	auto gsigma = grads[1];

	auto u = raw_u;
	auto v = raw_v;
	auto gu = grads[0];
	auto gv = grads[2];
/*
	if (!some)
	{
		// We ignore the free subspace here because possible base vectors cancel
		// each other, e.g., both -v and +v are valid base for a dimension.
		// Don't assume behavior of any particular implementation of svd.
	    u = raw_u.narrow(1, 0, k);
	    v = raw_v.narrow(1, 0, k);
	    if (gu.defined())
	    {
	    	gu = gu.narrow(1, 0, k);
	    }
	    if (gv.defined())
	    {
	    	gv = gv.narrow(1, 0, k);
	    }
	}
*/
	auto vt = v.t();

	cv::Mat sigma_term;

	if (!gsigma.empty())
	{
		sigma_term = (u * cv::Mat::diag(gsigma)) * vt;
	}
	else
	{
		sigma_term = cv::Mat::zeros(self.size(), self.type());
	}
	// in case that there are no gu and gv, we can avoid the series of kernel
	// calls below
	if (gv.empty() && gu.empty())
	{
		return sigma_term;
	}

	auto ut = u.t();
	auto im = cv::Mat::eye((int)m, (int)m, self.type());
	auto in = cv::Mat::eye((int)n, (int)n, self.type());
	auto sigma_mat = cv::Mat::diag(sigma);

	cv::Mat sigma_mat_inv;
	cv::pow(sigma, -1, sigma_mat_inv);
	sigma_mat_inv = cv::Mat::diag(sigma_mat_inv);

	cv::Mat sigma_sq, sigma_expanded_sq;
	cv::pow(sigma, 2, sigma_sq);
	sigma_expanded_sq = cv::repeat(sigma_sq, sigma_mat.rows, 1);

	cv::Mat F = sigma_expanded_sq - sigma_expanded_sq.t();
	// The following two lines invert values of F, and fills the diagonal with 0s.
	// Notice that F currently has 0s on diagonal. So we fill diagonal with +inf
	// first to prevent nan from appearing in backward of this function.
	F.diag().setTo(std::numeric_limits<float>::max());
	cv::pow(F, -1, F);

	cv::Mat u_term, v_term;

	if (!gu.empty())
	{
		cv::multiply(F, ut*gu-gu.t()*u, u_term);
		u_term = (u * u_term) * sigma_mat;
		if (m > k)
		{
			u_term = u_term + ((im - u*ut)*gu)*sigma_mat_inv;
		}
		u_term = u_term * vt;
	}
	else
	{
		u_term = cv::Mat::zeros(self.size(), self.type());
	}

	if (!gv.empty())
	{
		auto gvt = gv.t();
		cv::multiply(F, vt*gv - gvt*v, v_term);
		v_term = (sigma_mat*v_term) * vt;
		if (n > k)
		{
			v_term = v_term + sigma_mat_inv*(gvt*(in - v*vt));
		}
		v_term = u * v_term;
	}
	else
	{
		v_term = cv::Mat::zeros(self.size(), self.type());
	}

	return u_term + sigma_term + v_term;
}

bool kabsch(std::vector<cv::Point3f>& measurePts, std::vector<cv::Point3f>& scenePts, cv::Mat& r, cv::Mat& t, cv::Mat& jacobian, bool calc=false)
{

	cv::Mat P, X, A, U, W, Vt, D, R, V;
	cv::Mat gRodr;

	P = cv::Mat(measurePts, true).reshape(1, measurePts.size());

	X = cv::Mat(scenePts, true).reshape(1, scenePts.size());

	cv::Mat tx = (cv::Mat_<float>(1, 3) <<
					cv::mean(X.colRange(0,1))[0],
					cv::mean(X.colRange(1,2))[0],
					cv::mean(X.colRange(2,3))[0]);
	cv::Mat tp = (cv::Mat_<float>(1, 3) <<
					cv::mean(P.colRange(0,1))[0],
					cv::mean(P.colRange(1,2))[0],
					cv::mean(P.colRange(2,3))[0]);

	cv::Mat Xc = X - cv::repeat(tx, scenePts.size(), 1);
	cv::Mat Pc = P - cv::repeat(tp, measurePts.size(), 1);

	A = Pc.t() * Xc;

	cv::SVD::compute(A, W, U, Vt);

	float d = cv::determinant(U * Vt);

	D = (cv::Mat_<float>(3,3) <<
				1., 0., 0.,
				0., 1., 0.,
				0., 0., d );

	R = U * (D * Vt);

	if (calc)
	{
		cv::Rodrigues(R, r, gRodr);
	}
	else
	{
		cv::Rodrigues(R, r);
	}


	t = tp - (R*tx.t()).t();

	r = r.reshape(1, 3);
	t = t.reshape(1,3);

	if (!calc)
		return true;

	cv::Mat onehot;
	cv::Mat jac, drdR, dRdU, dRdVt, drdU, drdVt, drdV, drdA, dAdPct, dAdXc, drdXc;

	jac = cv::Mat_<double>::zeros(6, scenePts.size() * 3);
	cv::matMulDeriv(U, Vt, dRdU, dRdVt);
	cv::matMulDeriv(Pc.t(), Xc, dAdPct, dAdXc);

	dRdU = dRdU.t();
	dRdVt = dRdVt.t();
	dAdPct = dAdPct.t();
	dAdXc = dAdXc.t();

//	print("dAdXc", dAdXc);

	V = Vt.t();
	W = W.reshape(1, 1);

	for (unsigned int i = 0; i < r.rows; ++i)
	{

		onehot = cv::Mat::zeros(3, 1, CV_32F);
		onehot.at<float>(i, 0) = 1;

		drdR = gRodr * onehot;

		drdU = dRdU * drdR;
		drdVt = dRdVt * drdR;

		drdU = drdU.reshape(1, 3);
		drdVt = drdVt.reshape(1, 3);

		drdV = drdVt.t();

		std::vector<cv::Mat> grads{drdU, cv::Mat(), drdVt};

		drdA = svd_backward(grads, A, true, U, W, Vt);

		drdA = drdA.reshape(1, 9);

		drdXc = dAdXc * drdA;

		drdXc = drdXc.reshape(1, 1);

		drdXc.copyTo(jac.rowRange(i, i+1));

	}

	print("jacobian", jac);

	return true;

}



void kabsch_backward()
{

}

cv::Mat_<double> kabsch_fd(std::vector<cv::Point3f>& imgdPts, std::vector<cv::Point3f> objPts, float eps = 0.001f)
{
	cv::Mat_<double> jacobean = cv::Mat_<double>::zeros(6, objPts.size()*3);
	bool success;

	for (unsigned int i = 0; i < objPts.size(); ++i)
	{
		for (unsigned int j = 0; j < 3; ++j)
		{

			if(j == 0) objPts[i].x += eps;
			else if(j == 1) objPts[i].y += eps;
			else if(j == 2) objPts[i].z += eps;

			// forward step

			jp::cv_trans_t fStep; cv::Mat jac;
			success = kabsch(imgdPts, objPts, fStep.first, fStep.second, jac); //return success flag?

			if(!success)
				return cv::Mat_<double>::zeros(6, objPts.size() * 3);

			if(j == 0) objPts[i].x -= 2 * eps;
			else if(j == 1) objPts[i].y -= 2 * eps;
			else if(j == 2) objPts[i].z -= 2 * eps;

			// backward step
			jp::cv_trans_t bStep;
			success = kabsch(imgdPts, objPts, bStep.first, bStep.second, jac); //return success flag?

			if(!success)
				return cv::Mat_<double>::zeros(6, objPts.size() * 3);

			if(j == 0) objPts[i].x += eps;
			else if(j == 1) objPts[i].y += eps;
			else if(j == 2) objPts[i].z += eps;

			// gradient calculation
			fStep.first = (fStep.first - bStep.first) / (2 * eps);
			fStep.second = (fStep.second - bStep.second) / (2 * eps);

			fStep.first.copyTo(jacobean.col(i * 3 + j).rowRange(0, 3));
			fStep.second.copyTo(jacobean.col(i * 3 + j).rowRange(3, 6));

			if(containsNaNs(jacobean.col(i * 3 + j)))
				return cv::Mat_<double>::zeros(6, objPts.size() * 3);

		}
	}

	return jacobean;

}





/*
 * Test functions
 */
void test_accuracy()
{
	std::vector<unsigned int> powers {2};
	std::vector<unsigned int> N;
	unsigned int trials = 1;
	float etol = 1e-3;
	float delta = 0.001f;

	unsigned int cntErrJac = 0, cntErrRot = 0, cntErrTrans = 0, cntErrRotFD = 0, cntDegen = 0;
	unsigned int sumErrJac = 0, sumErrRot = 0, sumErrTrans = 0, sumErrRotFD = 0;

	cv::Vec3f angles = cv::Vec3f(0., M_PI/2., M_PI/3.);
	cv::Vec3f trans = cv::Vec3f(0., 0., 0.);
	Transformation tf(&angles, &trans);

	std::cout << "test translation:" << std::endl << cv::repeat(cv::Mat(tf.getTranslation()).reshape(1, 1), 3, 1) << std::endl;

	std::cout << "Rotation matrix = " << std::endl << " " << tf.getRotationMatrix() << std::endl;
	std::cout << "Translation = " << std::endl << " " << tf.getTranslation() << std::endl;
	std::cout << "Rotation vector = " << std::endl << " " << tf.getRotationVector() << std::endl;
	std::cout << "Rotation angles = " << std::endl << " " << tf.getRotationAngles() << std::endl;
	std::vector<cv::Point3f> scenePts = createScene(4);

	std::cout << "Data = " << std::endl << " " << scenePts << std::endl;

	std::vector<cv::Point3f> measurePts = createMeasurements(scenePts, tf);

	std::cout << "Mesurements = " << std::endl << " " << measurePts << std::endl;

	cv::Mat r_est, t_est, jacobian;

	kabsch(measurePts, scenePts, r_est, t_est, jacobian, true);

	std::cout << "Estimated rotation vector = " << std::endl << " " << r_est << std::endl;
	std::cout << "Estimated translation vector = " << std::endl << " " << t_est << std::endl;

//	cv::Mat jacFd = kabsch_fd(measurePts, scenePts);

//	std::cout << "Jacobian (finite difference): " << std::endl << " " << jacFd << std::endl;
}

void test_svd_backward()
{
	cv::Mat gU = (cv::Mat_<float>(3,3) <<
					0., 0., 0.,
					0., 0., 0.7854,
					0.5554, -0.5554, 0.);
	cv::Mat gS;

	cv::Mat gV = (cv::Mat_<float>(3,3) <<
					0., 0., 0.,
					0.5554, 0.5554, 0.,
					-0.5554, 0.5554, 0.);

	cv::Mat A = (cv::Mat_<float>(3,3) <<
					0., 0., 0.,
					1., 2., 0.,
					2., 1., 0.);

	cv::Mat U = (cv::Mat_<float>(3,3) <<
					0., 0., 1.,
					0.7071, -0.7071, 0.,
					0.7071, 0.7071, 0. );

	cv::Mat S = (cv::Mat_<float>(1,3) <<
					3., 1., 0.);

	cv::Mat V = (cv::Mat_<float>(3,3) <<
					0.7071, 0.7071, 0.,
					0.7071, -0.7071, 0.,
					0., 0., 1. );

	std::vector<cv::Mat> grads{gU, gS, gV};

	cv::Mat res = svd_backward(grads, A, true, U, S, V);

	std::cout << "result: " << std::endl << " " << res << std::endl;
}

void test_mm_backward()
{
	cv::Mat P = (cv::Mat_<float>(3,4) <<
				2., 2., 2., 2.,
				5., 5., 5., 5.,
				7., 7., 7., 7.);

	cv::Mat X = (cv::Mat_<float>(4,3) <<
					12., 12., 12.,
					15., 15., 15.,
					17., 17., 17.,
					13., 13., 13.);

	cv::Mat dA, dB;

	cv::matMulDeriv(P, X, dA, dB);

	print("dA", dA);
	print("dB", dB);
}

int main(int argc, char** argv)
{

	std::cout << "Kabsch Algorithm..." << std::endl;

	test_accuracy();

//	test_svd_backward();

//	test_mm_backward();

//	float data[9] = {-0.2160, -0.6887,  0.0000, -0.7114,  0.5965,  0.0000, -0.6687, -0.4121,  0.0000};
//	cv::Mat M(3,3, CV_32F, data);
//	std::cout << "M = " << std::endl << " " << M << std::endl << std::endl;
//
//	// OpenCV implementation
//	cv::Mat u, w, vt;
//	std::cout << "[start] OpenCV SVD decomposition" << std::endl;
//	cv::SVD::compute(M, w, u, vt);
//	std::cout << "[end] OpenCV SVD decomposition" << std::endl;
//	std::cout << "u = " << std::endl << " " << u << std::endl << std::endl;
//	std::cout << "w = " << std::endl << " " << w << std::endl << std::endl;
//	std::cout << "vt = " << std::endl << " " << vt << std::endl << std::endl;

	return 0;
}
