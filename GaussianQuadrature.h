#pragma once

#include <Eigen/Dense>
#include <stdexcept>

class GaussianQuadrature {
public:
    static void ComputeNodesAndWeights(int number_of_points,
        double lower_limit,
        double upper_limit,
        Eigen::VectorXd& integration_nodes,
        Eigen::VectorXd& integration_weights);

private:
    static void ComputeHighOrderNodesAndWeights(int number_of_points,
        Eigen::VectorXd& integration_nodes,
        Eigen::VectorXd& integration_weights);

    static void TransformToTargetInterval(double interval_start,
        double interval_end,
        Eigen::VectorXd& integration_nodes,
        Eigen::VectorXd& integration_weights);
};
