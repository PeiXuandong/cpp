#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <functional>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class BasisFunction {
public:
    // 计算有限元形状函数及其导数
    static MatrixXd FEMShapeFunction(const VectorXd& x_points,
        int degree,
        int derivative_order)
    {
        int node_count = degree + 1;
        VectorXd node_positions = VectorXd::LinSpaced(node_count, 0.0, 1.0);


        // 计算系数矩阵
        MatrixXd coefficient_matrix = MatrixXd::Zero(node_count - 1, node_count);
        for (int i = 0; i < node_count; ++i) {
            int index = 0;
            for (int j = 0; j < node_count; ++j) {
                if (i != j) {
                    coefficient_matrix(index, i) = node_positions(i) - node_positions(j);
                    index++;
                }
            }
        }

        // 计算系数的倒数
        VectorXd coefficient_inverse(node_count);
        for (int i = 0; i < node_count; ++i) {
            double product = 1.0;
            for (int j = 0; j < node_count; ++j) {
                if (j != i) {
                    product *= (node_positions(i) - node_positions(j));
                }
            }
            coefficient_inverse(i) = 1.0 / product;
        }

        // 初始化结果矩阵
        MatrixXd shape_function_values(x_points.size(), node_count);

        if (derivative_order == 0) {
            // 计算形状函数值
            for (int i = 0; i < node_count; ++i) {
                for (int point_index = 0; point_index < x_points.size(); ++point_index) {
                    double product = 1.0;
                    for (int j = 0; j < node_count; ++j) {
                        if (j != i) {
                            product *= (x_points(point_index) - node_positions(j));
                        }
                    }
                    shape_function_values(point_index, i) = product * coefficient_inverse(i);
                }
            }
        }
        else if (derivative_order == 1) {
            // 计算一阶导数
            for (int i = 0; i < node_count; ++i) {
                for (int point_index = 0; point_index < x_points.size(); ++point_index) {
                    double sum = 0.0;
                    for (int skip_index = 0; skip_index < node_count; ++skip_index) {
                        if (skip_index != i) {
                            double product = 1.0;
                            for (int j = 0; j < node_count; ++j) {
                                if (j != i && j != skip_index) {
                                    product *= (x_points(point_index) - node_positions(j));
                                }
                            }
                            sum += product;
                        }
                    }
                    shape_function_values(point_index, i) = sum * coefficient_inverse(i);
                }
            }
        }
        else {
            throw runtime_error("Higher derivatives have not been implemented!");
        }

        return shape_function_values;
    }

    // 计算有限元形状函数在全局坐标下的值
    static MatrixXd FEMEval(const VectorXd& evaluation_points,
        double interval_start,
        double interval_end,
        int element_count)
    {
        auto f1 = [](double x) { return -0.5 * (3 * x - 1) * (3 * x - 2) * (x - 1); };
        auto f2 = [](double x) { return 4.5 * x * (3 * x - 2) * (x - 1); };
        auto f3 = [](double x) { return -4.5 * x * (3 * x - 1) * (x - 1); };
        auto f4 = [](double x) { return 0.5 * x * (3 * x - 1) * (3 * x - 2); };

        double element_length = (interval_end - interval_start) / element_count;
        MatrixXd function_values(3 * element_count + 1, evaluation_points.size());
        function_values.setZero();

        for (int point_index = 0; point_index < evaluation_points.size(); ++point_index) {
            double local_x = fmod(evaluation_points(point_index) - interval_start, element_length) / element_length;
            int element_index = static_cast<int>((evaluation_points(point_index) - interval_start) / element_length) + 1;

            if (element_index <= element_count) {
                function_values(3 * element_index - 2, point_index) = f1(local_x);
                function_values(3 * element_index - 1, point_index) = f2(local_x);
                function_values(3 * element_index, point_index) = f3(local_x);
                function_values(3 * element_index + 1, point_index) = f4(local_x);
            }
            else {
                function_values(3 * element_count + 1, point_index) = f1(local_x);
            }
        }

        return function_values;
    }


    // 计算单位分解形状函数及其导数
    static MatrixXd PUShapeFunction(const VectorXd& x_points,
        int derivative_order)
    {
        MatrixXd function_values(x_points.size(), 2);

        if (derivative_order == 0) {
            for (int i = 0; i < x_points.size(); ++i) {
                double x = x_points(i);
                function_values(i, 0) = pow(1 - x, 2) * (1 + 2 * x);
                function_values(i, 1) = x * x * (3 - 2 * x);
            }
        }
        else if (derivative_order == 1) {
            for (int i = 0; i < x_points.size(); ++i) {
                double x = x_points(i);
                function_values(i, 0) = pow(1 - x, 2) * 2 - 2 * (1 - x) * (1 + 2 * x);
                function_values(i, 1) = 2 * x * (3 - 2 * x) - 2 * x * x;
            }
        }
        else {
            throw runtime_error("Higher derivatives have not been implemented!");
        }

        return function_values;
    }

    // 计算单位分解形状函数在全局坐标下的值
    static MatrixXd PUEval(const VectorXd& evaluation_points,
        double interval_start,
        double interval_end,
        int element_count)
    {
        auto f1 = [](double x) { return pow(1 - x, 2) * (1 + 2 * x); };
        auto f2 = [](double x) { return x * x * (3 - 2 * x); };

        double element_length = (interval_end - interval_start) / element_count;
        MatrixXd function_values(element_count + 1, evaluation_points.size());
        function_values.setZero();

        for (int point_index = 0; point_index < evaluation_points.size(); ++point_index) {
            double local_x = fmod(evaluation_points(point_index) - interval_start, element_length) / element_length;
            int element_index = static_cast<int>((evaluation_points(point_index) - interval_start) / element_length) + 1;

            if (element_index <= element_count) {
                function_values(element_index, point_index) = f1(local_x);
                function_values(element_index + 1, point_index) = f2(local_x);
            }
            else {
                function_values(element_count + 1, point_index) = f1(local_x);
            }
        }

        return function_values;
    }

    // 计算单位分解形状函数的导数在全局坐标下的值
    static MatrixXd DPUEval(const VectorXd& evaluation_points,
        double interval_start,
        double interval_end,
        int element_count)
    {
        auto f1 = [](double x) { return pow(1 - x, 2) * 2 - 2 * (1 - x) * (1 + 2 * x); };
        auto f2 = [](double x) { return 2 * x * (3 - 2 * x) - 2 * x * x; };

        double element_length = (interval_end - interval_start) / element_count;
        MatrixXd derivative_values(element_count + 1, evaluation_points.size());
        derivative_values.setZero();

        for (int point_index = 0; point_index < evaluation_points.size(); ++point_index) {
            double local_x = fmod(evaluation_points(point_index) - interval_start, element_length) / element_length;
            int element_index = static_cast<int>((evaluation_points(point_index) - interval_start) / element_length) + 1;

            if (element_index <= element_count) {
                derivative_values(element_index, point_index) = f1(local_x) / element_length;
                derivative_values(element_index + 1, point_index) = f2(local_x) / element_length;
            }
            else {
                derivative_values(element_count + 1, point_index) = f1(local_x) / element_length;
            }
        }

        return derivative_values;
    }
};


