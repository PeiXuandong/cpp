#include <cmath>
#include <functional>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Constants

// Interface function (to be implemented)
double intface(double x) { return tan(M_PI / 6.0) * (x - 1 - 1.0 / M_PI) + 1; }
double intfacex(double x) { return tan(M_PI / 6.0) * (x - 1 - 1.0 / M_PI) + 1; }
double intfacey(double y) { return (y - 1.0) / tan(M_PI / 6.0) + 1 + 1.0 / M_PI; }

constexpr double pi = 3.14159265358979323846;
constexpr double a0 = 1.0;  // 设为你的实际值
constexpr double a1 = 10.0;  // 设为你的实际值

// Polar coordinate evaluation function
template<typename Functor>
double Ueval(Functor&& fun, double x, double y) {

    double x0 = 1.0 + 1.0 / pi;
    double y0 = 1.0;


    double dx = x - x0;
    double dy = y - y0;
    double r = sqrt(dx * dx + dy * dy);
    double theta = atan2(dy, dx);

    if (theta > 0.0) {
        theta -= 2.0 * pi;
    }

    return fun(r, theta, x, y);
}


// Exact solutions
double exact_u(double r, double theta, double x, double y) {
    constexpr double pi = 3.14159265358979323846;
    constexpr double a0 = 1.0;  // 设为你的实际值
    constexpr double a1 = 10.0;  // 设为你的实际值

    if (y > intface(x)) {
        return pow(r, 4.0 / 3.0) * cos((4.0 / 3.0) * (theta + pi - pi / 6.0))
            + (a0 / a1) * pow(r, 4.0 / 3.0) * sin((4.0 / 3.0) * (theta + pi - pi / 6.0))
            + sin(x * y);
    }
    else {
        return pow(r, 4.0 / 3.0) * cos((4.0 / 3.0) * (theta + pi - pi / 6.0))
            + pow(r, 4.0 / 3.0) * sin((4.0 / 3.0) * (theta + pi - pi / 6.0))
            + sin(x * y);
    }
}

double exact_f(double r, double theta, double x, double y) {
    if (y > intface(x)) {
        return a1 * (x * x + y * y) * sin(x * y);
    }
    else {
        return a0 * (x * x + y * y) * sin(x * y);
    }
}

// Derivatives
double exact_ux(double r, double theta, double x, double y) {
    if (y > intface(x)) {
        return -(4 * pow(r, 1.0 / 3.0) * (a1 * cos(theta / 3.0 + pi / 9.0) + a0 * sin(theta / 3.0 + pi / 9.0))) / (3.0 * a1)
            + y * cos(x * y);
    }
    else {
        return -(4 * pow(2, 1.0 / 2.0) * pow(r, 1.0 / 3.0) * cos(theta / 3.0 - (5 * pi) / 36.0)) / 3.0
            + y * cos(x * y);
    }
}

double exact_uy(double r, double theta, double x, double y) {
    if (y > intface(x)) {
        return -(4 * pow(r, 1.0 / 3.0) * (a0 * cos(theta / 3.0 + pi / 9.0) - a1 * sin(theta / 3.0 + pi / 9.0))) / (3.0 * a1)
            + x * cos(x * y);
    }
    else {
        return -(4 * pow(2, 1.0 / 2.0) * pow(r, 1.0 / 3.0) * cos(theta / 3.0 + (13 * pi) / 36.0)) / 3.0
            + x * cos(x * y);
    }
}

// Wrapper functions
double calc_ux_up(double x, double y) {
    auto fun = [](double r, double theta, double x, double y) {
        return -(4 * pow(r, 1.0 / 3.0) * (a1 * cos(theta / 3.0 + pi / 9.0) + a0 * sin(theta / 3.0 + pi / 9.0))) / (3.0 * a1)
            + y * cos(x * y);
        };
    return Ueval(fun, x, y);
}

double calc_ux_bel(double x, double y) {
    auto fun = [](double r, double theta, double x, double y) {
        return -(4 * pow(2, 1.0 / 2.0) * pow(r, 1.0 / 3.0) * cos(theta / 3.0 - (5 * pi) / 36.0)) / 3.0
            + y * cos(x * y);
        };
    return Ueval(fun, x, y);
}

double calc_uy_up(double x, double y) {
    auto fun = [](double r, double theta, double x, double y) {
        return -(4 * pow(r, 1.0 / 3.0) * (a0 * cos(theta / 3.0 + pi / 9.0) - a1 * sin(theta / 3.0 + pi / 9.0))) / (3.0 * a1)
            + x * cos(x * y);
        };
    return Ueval(fun, x, y);
}

double calc_uy_bel(double x, double y) {
    auto fun = [](double r, double theta, double x, double y) {
        return -(4 * pow(2, 1.0 / 2.0) * pow(r, 1.0 / 3.0) * cos(theta / 3.0 + (13 * pi) / 36.0)) / 3.0
            + x * cos(x * y);
        };
    return Ueval(fun, x, y);
}

// Flux calculation
double calc_q(double x, double y, const vector<double>& n0, const vector<double>& n1) {
    double ux_up = calc_ux_up(x, y);
    double uy_up = calc_uy_up(x, y);
    double ux_bel = calc_ux_bel(x, y);
    double uy_bel = calc_uy_bel(x, y);

    return a1 * (ux_up * n0[0] + uy_up * n0[1]) + a0 * (ux_bel * n1[0] + uy_bel * n1[1]);
}

// Main calculation functions
double calc_exact_u(double x, double y) {
    auto fun = [](double r, double theta, double x, double y) {
        return exact_u(r, theta, x, y);
        };
    return Ueval(fun, x, y);
}

double calc_exact_f(double x, double y) {
    auto fun = [](double r, double theta, double x, double y) {
        return exact_f(r, theta, x, y);
        };
    return Ueval(fun, x, y);
}

double calc_exact_ux(double x, double y) {
    auto fun = [](double r, double theta, double x, double y) {
        return exact_ux(r, theta, x, y);
        };
    return Ueval(fun, x, y);
}

double calc_exact_uy(double x, double y) {
    auto fun = [](double r, double theta, double x, double y) {
        return exact_uy(r, theta, x, y);
        };
    return Ueval(fun, x, y);
}

// 计算全局点的函数值 (对应MATLAB的 val_f = calc_exact_f(globalGps(:, 1), globalGps(:, 2)))
VectorXd compute_val_f(const MatrixXd& globalGps) {
    VectorXd val_f(globalGps.rows());
    for (int i = 0; i < globalGps.rows(); ++i) {
        val_f(i) = calc_exact_f(globalGps(i, 0), globalGps(i, 1));
    }
    return val_f;
}


void mergeGpsValues(MatrixXd& valuelist, const MatrixXd& gpsValues, const VectorXi& posF) {
    if (valuelist.rows() == 0) {
        valuelist = gpsValues(all, posF);
    }
    else {
        MatrixXd V(gpsValues.rows() + valuelist.rows(), posF.size());
        V << valuelist, gpsValues(all, posF);
        valuelist = V;
    }
}
