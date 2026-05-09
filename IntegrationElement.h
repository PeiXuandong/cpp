#pragma once
#include <vector>
#include <array>
#include <Eigen/Dense>
#include <tuple>
#include "GaussianQuadrature.h"

using namespace Eigen;
using namespace std;
// At the top of your file, add:
typedef Matrix<double, 1, 5> RowVector5d;


class IntegrationElement {
private:
    const int quad_nqps = 25;
    const int tri_nqps = 25;

    // Helper functions
    tuple<MatrixXd, VectorXd, MatrixXd> TriangleGaussPoint(const VectorXd& x, const VectorXd& y, int n);
    tuple<MatrixXd, VectorXd, MatrixXd> QuadGaussPoint(const VectorXd& x, const VectorXd& y, int n);
pair<MatrixXd, MatrixXd> intElemIntegration_Line(const array<array<bool, 2>, 2>& C, int i, int j, int N, const VectorXd& x, const VectorXd& y);private:
    // Add this inside the private section of your IntegrationElement class
    tuple<MatrixXd, VectorXd> TriangleGauss2(int n);

public:
    struct IntegrationResult {
        MatrixXd Gps0;
        VectorXd GpW0;
        MatrixXd Jac0;
        MatrixXd Gps1;
        VectorXd GpW1;
        MatrixXd Jac1;
    };

    IntegrationResult intElemIntegration(const array<array<bool, 2>, 2>& C, int i, int j, int N, const VectorXd& xcoord, const VectorXd& ycoord);
};

IntegrationElement::IntegrationResult IntegrationElement::intElemIntegration(
    const array<array<bool, 2>, 2>& C, int i, int j, int N,
    const VectorXd& xcoord, const VectorXd& ycoord) {

    IntegrationResult result;

    auto  result_line = intElemIntegration_Line(C, i, j, N, xcoord, ycoord);
    MatrixXd part0 = result_line.first;
    MatrixXd part1 = result_line.second;


    // Process part0
    if (part0.cols() == 3) {
        tie(result.Gps0, result.GpW0, result.Jac0) = TriangleGaussPoint(part0.row(0).transpose(), part0.row(1).transpose(), tri_nqps);
    }
    else if (part0.cols() == 4) {
        tie(result.Gps0, result.GpW0, result.Jac0) = QuadGaussPoint(part0.row(0).transpose(), part0.row(1).transpose(), quad_nqps);
    }
    else if (part0.cols() == 5) {
        MatrixXd part0_quad = part0.block(0, 0, 2, 4);
        MatrixXd part0_tri(2, 3);
        part0_tri << part0.col(3), part0.col(4), part0.col(0);

        tie(result.Gps0, result.GpW0, result.Jac0) = QuadGaussPoint(part0_quad.row(0).transpose(), part0_quad.row(1).transpose(), quad_nqps);

        MatrixXd TempGps;
        VectorXd TempGpW;
        MatrixXd TempJac;
        tie(TempGps, TempGpW, TempJac) = TriangleGaussPoint(part0_tri.row(0).transpose(), part0_tri.row(1).transpose(), tri_nqps);

        result.Gps0 = (MatrixXd(result.Gps0.rows() + TempGps.rows(), 2) << result.Gps0, TempGps).finished();
        result.GpW0 = (VectorXd(result.GpW0.size() + TempGpW.size()) << result.GpW0, TempGpW).finished();
        result.Jac0 = (MatrixXd(result.Jac0.rows() + TempJac.rows(), 5) << result.Jac0, TempJac).finished();
    }

    // Process part1
    if (part1.cols() == 3) {
        tie(result.Gps1, result.GpW1, result.Jac1) = TriangleGaussPoint(part1.row(0).transpose(), part1.row(1).transpose(), tri_nqps);
    }
    else if (part1.cols() == 4) {
        tie(result.Gps1, result.GpW1, result.Jac1) = QuadGaussPoint(part1.row(0).transpose(), part1.row(1).transpose(), quad_nqps);
    }
    else if (part1.cols() == 5) {
        MatrixXd part1_quad = part1.block(0, 0, 2, 4);
        MatrixXd part1_tri(2, 3);
        part1_tri << part1.col(3), part1.col(4), part1.col(0);

        tie(result.Gps1, result.GpW1, result.Jac1) = QuadGaussPoint(part1_quad.row(0).transpose(), part1_quad.row(1).transpose(), quad_nqps);

        MatrixXd TempGps;
        VectorXd TempGpW;
        MatrixXd TempJac;
        tie(TempGps, TempGpW, TempJac) = TriangleGaussPoint(part1_tri.row(0).transpose(), part1_tri.row(1).transpose(), tri_nqps);

        result.Gps1 = (MatrixXd(result.Gps1.rows() + TempGps.rows(), 2) << result.Gps1, TempGps).finished();
        result.GpW1 = (VectorXd(result.GpW1.size() + TempGpW.size()) << result.GpW1, TempGpW).finished();
        result.Jac1 = (MatrixXd(result.Jac1.rows() + TempJac.rows(), 5) << result.Jac1, TempJac).finished();
    }

    return result;
}

pair<MatrixXd, MatrixXd> IntegrationElement::intElemIntegration_Line(
    const array<array<bool, 2>, 2>& C, int i, int j, int N,
    const VectorXd& x, const VectorXd& y) {

    vector<double> part0x, part0y, part1x, part1y;

    double h = 1.0 / N;
    double x0 = (j - 1) * h;
    double y0 = (i - 1) * h;
    bool flag = false;
    int iter = 0;

    if (!C[0][0]) {
        part0x.push_back(x0);
        part0y.push_back(y0);
        flag = false;
    }
    else {
        part1x.push_back(x0);
        part1y.push_back(y0);
        flag = true;
    }

    if (C[0][1] == C[0][0]) {
        if (!flag) {
            part0x.push_back(x0 + h);
            part0y.push_back(y0);
        }
        else {
            part1x.push_back(x0 + h);
            part1y.push_back(y0);
        }
    }
    else {
        if (!flag) {
            part0x.push_back(x(iter));
            part0y.push_back(y(iter));
            part1x.push_back(x(iter));
            part1y.push_back(y(iter));
            part1x.push_back(x0 + h);
            part1y.push_back(y0);
            iter++;
            flag = !flag;
        }
        else {
            part1x.push_back(x(iter));
            part1y.push_back(y(iter));
            part0x.push_back(x(iter));
            part0y.push_back(y(iter));
            part0x.push_back(x0 + h);
            part0y.push_back(y0);
            iter++;
            flag = !flag;
        }
    }

    if (C[1][1] == C[0][1]) {
        if (!flag) {
            part0x.push_back(x0 + h);
            part0y.push_back(y0 + h);
        }
        else {
            part1x.push_back(x0 + h);
            part1y.push_back(y0 + h);
        }
    }
    else {
        if (!flag) {
            part0x.push_back(x(iter));
            part0y.push_back(y(iter));
            part1x.push_back(x(iter));
            part1y.push_back(y(iter));
            part1x.push_back(x0 + h);
            part1y.push_back(y0 + h);
            iter++;
            flag = !flag;
        }
        else {
            part1x.push_back(x(iter));
            part1y.push_back(y(iter));
            part0x.push_back(x(iter));
            part0y.push_back(y(iter));
            part0x.push_back(x0 + h);
            part0y.push_back(y0 + h);
            iter++;
            flag = !flag;
        }
    }

    if (C[1][0] == C[1][1]) {
        if (!flag) {
            part0x.push_back(x0);
            part0y.push_back(y0 + h);
        }
        else {
            part1x.push_back(x0);
            part1y.push_back(y0 + h);
        }
    }
    else {
        if (!flag) {
            part0x.push_back(x(iter));
            part0y.push_back(y(iter));
            part1x.push_back(x(iter));
            part1y.push_back(y(iter));
            part1x.push_back(x0);
            part1y.push_back(y0 + h);
            iter++;
            flag = !flag;
        }
        else {
            part1x.push_back(x(iter));
            part1y.push_back(y(iter));
            part0x.push_back(x(iter));
            part0y.push_back(y(iter));
            part0x.push_back(x0);
            part0y.push_back(y0 + h);
            iter++;
            flag = !flag;
        }
    }

    if (C[0][0] != C[1][0]) {
        part0x.push_back(x(iter));
        part0y.push_back(y(iter));
        part1x.push_back(x(iter));
        part1y.push_back(y(iter));
    }

    MatrixXd part0(2, part0x.size());
    part0.row(0) = Map<VectorXd>(part0x.data(), part0x.size());
    part0.row(1) = Map<VectorXd>(part0y.data(), part0y.size());

    MatrixXd part1(2, part1x.size());
    part1.row(0) = Map<VectorXd>(part1x.data(), part1x.size());
    part1.row(1) = Map<VectorXd>(part1y.data(), part1y.size());

    return make_pair(part0, part1);
}

tuple<MatrixXd, VectorXd, MatrixXd> IntegrationElement::TriangleGaussPoint(
    const VectorXd& x, const VectorXd& y, int n) {

    // You'll need to implement TriangleGauss2(n) equivalent in C++
    auto result_triangle = TriangleGauss2(n);
    MatrixXd reg_gps = get<0>(result_triangle);
    VectorXd GpW = get<1>(result_triangle);

    MatrixXd Jac = MatrixXd::Ones(GpW.size(), 5);
    Matrix2d mat;
    mat << x(1) - x(0), x(2) - x(0),
        y(1) - y(0), y(2) - y(0);

    MatrixXd Gps = reg_gps * mat.transpose();
    Gps.rowwise() += RowVector2d(x(0), y(0));

    double det_mat = mat.determinant();
    RowVector5d jac_row;
    jac_row << det_mat,
        (y(2) - y(0)) / det_mat, (x(0) - x(2)) / det_mat,
        (y(0) - y(1)) / det_mat, (x(1) - x(0)) / det_mat;

    Jac = jac_row.replicate(GpW.size(), 1);

    return make_tuple(Gps, GpW, Jac);
}

tuple<MatrixXd, VectorXd, MatrixXd> IntegrationElement::QuadGaussPoint(
    const VectorXd& x, const VectorXd& y, int n) {

    // You'll need to implement QuadGauss(n, 0, 1) equivalent in C++
   // auto [gps1d, gpw1d] = QuadGauss(n, 0, 1);
    VectorXd gps1d;
    VectorXd gpw1d;
    GaussianQuadrature::ComputeNodesAndWeights(n, 0.0, 1.0, gps1d, gpw1d);

    // Fix for gps1d replication
    // Generate 2D grid points
    MatrixXd reg_gps(gps1d.size() * gps1d.size(), 2);
    for (int i = 0; i < gps1d.size(); i++) {
        for (int j = 0; j < gps1d.size(); j++) {
            reg_gps(i * gps1d.size() + j, 0) = gps1d(i);
            reg_gps(i * gps1d.size() + j, 1) = gps1d(j);
        }
    }

    // Compute weights
    MatrixXd localGpW2D = gpw1d * gpw1d.transpose();
    VectorXd GpW = Map<VectorXd>(localGpW2D.data(), localGpW2D.size());

    // Compute Gauss points
    MatrixXd Gps(reg_gps.rows(), 2);
    for (int i = 0; i < reg_gps.rows(); i++) {
        double s = reg_gps(i, 0);
        double t = reg_gps(i, 1);

        // Bilinear mapping
        Gps(i, 0) = (1 - s) * (1 - t) * x(0) + s * (1 - t) * x(1) + s * t * x(2) + (1 - s) * t * x(3);
        Gps(i, 1) = (1 - s) * (1 - t) * y(0) + s * (1 - t) * y(1) + s * t * y(2) + (1 - s) * t * y(3);
    }

    // Compute Jacobian
    MatrixXd Jac(Gps.rows(), 5);
    for (int i = 0; i < reg_gps.rows(); i++) {
        double s = reg_gps(i, 0);
        double t = reg_gps(i, 1);

        // Derivatives of mapping
        double dx_ds = (1 - t) * (x(1) - x(0)) + t * (x(2) - x(3));
        double dy_ds = (1 - t) * (y(1) - y(0)) + t * (y(2) - y(3));
        double dx_dt = (1 - s) * (x(3) - x(0)) + s * (x(2) - x(1));
        double dy_dt = (1 - s) * (y(3) - y(0)) + s * (y(2) - y(1));

        double detJ = dx_ds * dy_dt - dx_dt * dy_ds;

        // Store Jacobian information
        Jac(i, 0) = detJ;
        Jac(i, 1) = dy_dt / detJ;   // ds_dx
        Jac(i, 2) = -dx_dt / detJ;  // ds_dy
        Jac(i, 3) = -dy_ds / detJ;  // dt_dx
        Jac(i, 4) = dx_ds / detJ;   // dt_dy
    }

    return make_tuple(Gps, GpW, Jac);
}

tuple<MatrixXd, VectorXd> IntegrationElement::TriangleGauss2(int n) {
    VectorXd gauss, w;
    GaussianQuadrature::ComputeNodesAndWeights(n, 0.0, 1.0, gauss, w);

    // Create s coordinates
    VectorXd s = gauss.replicate(n, 1);

    // Create t coordinates
    VectorXd t = gauss.transpose().replicate(n, 1).reshaped(n * n, 1);

    // Compute points
    MatrixXd points(n * n, 2);
    points.col(0) = s;
    points.col(1) = (1.0 - s.array()) * t.array();

    // Compute weights
    MatrixXd w_mat = w * w.transpose();
    VectorXd weights = w_mat.reshaped(n * n, 1).array() * (1.0 - s.array());

    return make_tuple(points, weights);
}
