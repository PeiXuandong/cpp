#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

int main() {
    std::cout << "Eigen版本信息:\n";
    std::cout << "================\n";
    
    // Eigen版本宏
    std::cout << "Eigen版本: " 
              << EIGEN_WORLD_VERSION << "."
              << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << std::endl;
    
    // 版本检查
    #if EIGEN_VERSION_AT_LEAST(3,4,0)
        std::cout << "✓ Eigen版本 >= 3.4.0，支持conditionNumberEstimate()" << std::endl;
    #else
        std::cout << "✗ Eigen版本 < 3.4.0，不支持conditionNumberEstimate()" << std::endl;
    #endif
    
    #if EIGEN_VERSION_AT_LEAST(3,3,0)
        std::cout << "✓ Eigen版本 >= 3.3.0，支持ConjugateGradient稳定版" << std::endl;
    #else
        std::cout << "✗ Eigen版本 < 3.3.0，可能有不稳定问题" << std::endl;
    #endif
    
    #if EIGEN_VERSION_AT_LEAST(3,2,0)
        std::cout << "✓ Eigen版本 >= 3.2.0，基础功能完整" << std::endl;
    #else
        std::cout << "✗ Eigen版本 < 3.2.0，建议升级" << std::endl;
    #endif
    
    // 额外信息
    std::cout << "\n编译信息:\n";
    std::cout << "EIGEN_WORLD_VERSION: " << EIGEN_WORLD_VERSION << std::endl;
    std::cout << "EIGEN_MAJOR_VERSION: " << EIGEN_MAJOR_VERSION << std::endl;
    std::cout << "EIGEN_MINOR_VERSION: " << EIGEN_MINOR_VERSION << std::endl;
    
    // 检查是否定义了其他版本信息
    #ifdef EIGEN_VERSION
        std::cout << "EIGEN_VERSION: " << EIGEN_VERSION << std::endl;
    #endif
    
    std::cout << "\n功能特性:\n";
    
    // 检查支持的求解器类型
    #ifdef EIGEN_USE_BLAS
        std::cout << "✓ BLAS支持已启用" << std::endl;
    #else
        std::cout << "- BLAS支持未启用" << std::endl;
    #endif
    
    #ifdef EIGEN_USE_LAPACKE
        std::cout << "✓ LAPACKE支持已启用" << std::endl;
    #else
        std::cout << "- LAPACKE支持未启用" << std::endl;
    #endif
    
    // 检查并行支持
    #ifdef EIGEN_HAS_OPENMP
        std::cout << "✓ OpenMP并行支持已启用" << std::endl;
    #else
        std::cout << "- OpenMP并行支持未启用" << std::endl;
    #endif
    
    #ifdef EIGEN_HAS_STD_THREAD
        std::cout << "✓ C++11线程支持已启用" << std::endl;
    #else
        std::cout << "- C++11线程支持未启用" << std::endl;
    #endif
    
    std::cout << "\n推荐升级建议:\n";
    
    #if EIGEN_VERSION_AT_LEAST(3,4,0)
        std::cout << "✓ 版本已是最新，无需升级" << std::endl;
    #elif EIGEN_VERSION_AT_LEAST(3,3,0)
        std::cout << "建议升级到3.4.0+以获得conditionNumberEstimate()支持" << std::endl;
    #elif EIGEN_VERSION_AT_LEAST(3,2,0)
        std::cout << "建议升级到3.3.0+以获得更好的稳定性和性能" << std::endl;
    #else
        std::cout << "强烈建议升级到3.4.0+，当前版本太旧" << std::endl;
    #endif
    
    return 0;
}
