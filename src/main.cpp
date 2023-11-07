// Description: main file for ugap_ga
//              optimization of material & process parameters
//              urethane grafted acrylate polymer (UGAP)
//              using genetic algorithm (GA)
// Created by Brian Howell on 10/31/23
// MSOL UC Berkeley
// bhowell@berkeley.edu
#include <iostream>
#include "Voxel.h"

void sort_data(Eigen::MatrixXd& PARAM);

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    // opt constraints
    constraints c; 

    // SINGLE SIMULATION
    bopt default_bopt;
    default_bopt.temp = 303.15;
    default_bopt.rp   = 0.00084 / 10;
    default_bopt.vp   = 0.7;
    default_bopt.uvi  = 10.;
    default_bopt.uvt  = 30.;
    
    sim default_sim;
    default_sim.time_stepping = 2;
    default_sim.update_time_stepping_values();
    const bool mthread = true; 
    int   save_voxel = 0;
    std::string file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/ugap_ga/output_" 
                            + std::to_string(default_sim.time_stepping);
    
    // GA parameters
    int pop = omp_get_num_procs();                                  // population size
    int P   = 4;                                                    // number of parents
    int C   = 4;                                                    // number of children
    int G   = 10;                                                   // number of generations
    double lam_1, lam_2;                                            // genetic alg paramters

    // || temp | rm | vp | uvi | uvt | obj || ∈ ℝ (pop x param + obj)
    Eigen::MatrixXd param(pop, 6);        

    // initialize input variables
    std::random_device rd;                                          // Obtain a random seed from the hardware
    std::mt19937 gen(rd());                                         // Seed the random number generator
    std::uniform_real_distribution<double> distribution(0.0, 1.0);  // Define the range [0.0, 1.0)

    // initials input samples
    std::cout << "--- INITIALIZING FIRST RUN ----" << std::endl;
    for (int i = 0; i < param.rows(); ++i){
        param(i, 0) = c.min_temp + (c.max_temp - c.min_temp) * distribution(gen);
        param(i, 1) = c.min_rp   + (c.max_rp   - c.min_rp)   * distribution(gen);
        param(i, 2) = c.min_vp   + (c.max_vp   - c.min_vp)   * distribution(gen);
        param(i, 3) = c.min_uvi  + (c.max_uvi  - c.min_uvi)  * distribution(gen);
        param(i, 4) = c.min_uvt  + (c.max_uvt  - c.min_uvt)  * distribution(gen);
    }

    // performance vectors
    std::vector<double> top_performer; 
    std::vector<double> avg_parent; 
    std::vector<double> avg_total;

    // initialize top performers
    #pragma omp parallel for
    for (int p = 0; p < pop; ++p) {
        Voxel sim(default_sim.tfinal,    // tot sim time
                  default_sim.dt,        // time step
                  default_sim.node,      // num nodes
                  default_sim.method,    // sim id
                  param(p, 0),           // amb temp
                  param(p, 3),           // uv intensity
                  param(p, 4),           // uv exposure time
                  default_sim.method,    // time stepping scheme
                  save_voxel,            // save voxel values
                  file_path,
                  mthread);
        sim.computeParticles(param(p, 1), param(p, 2));
        sim.simulate();
        #pragma omp critical
            {
                int thread_id = omp_get_thread_num();
                // std::cout << "Thread " << thread_id << std::endl;
                if (!std::isnan(sim.getObjective())) {
                    param(p, 5) = sim.getObjective();
                } else {
                    param(p, 5) = 1000.;
                }
            }
    }

    sort_data(param);

    // track top and average performers
    top_performer.push_back(param(0, param.cols() - 1));
    avg_parent.push_back(param.col(param.cols() - 1).head(P).mean());
    avg_total.push_back(param.col(param.cols() - 1).mean());
    

    std::cout << "\nparam: \n" << param << std::endl;

    // mate top performing samples
    for (int i = 0; i < P; i+=2) {
        // generate random numbers
        lam_1 = distribution(gen);
        lam_2 = distribution(gen);

        // skip parent rows and update child rows
        param.row(i + C)     = lam_1 * param.row(i) + (1-lam_1) * param.row(i+1);
        param.row(i + C + 1) = lam_2 * param.row(i) + (1-lam_2) * param.row(i+1);
        
        // reset obj
        param(i+C,   param.cols()-1) = 1000.0;
        param(i+C+1, param.cols()-1) = 1000.0;
    }

    // generate new pop for remaining rows
    for (int i = P + C; i < pop; ++i) {
        param(i, 0) = c.min_temp + (c.max_temp - c.min_temp) * distribution(gen);
        param(i, 1) = c.min_rp   + (c.max_rp   - c.min_rp)   * distribution(gen);
        param(i, 2) = c.min_vp   + (c.max_vp   - c.min_vp)   * distribution(gen);
        param(i, 3) = c.min_uvi  + (c.max_uvi  - c.min_uvi)  * distribution(gen);
        param(i, 4) = c.min_uvt  + (c.max_uvt  - c.min_uvt)  * distribution(gen); 
        param(i, 5) = 1000.0;
    }

    std::cout << "==== BEGIN GENERATIONS ====" << std::endl;
    // loop over generations
    for (int g = 0; g < G; ++g) {
        // loop over population 
        std::cout << "---- PARALLEZING POP " << g << " ----" << std::endl;
        #pragma omp parallel for
        for (int p = P; p < pop; ++p) {
            // initialize simulation
            Voxel sim(default_sim.tfinal,    // tot sim time
                      default_sim.dt,        // time step
                      default_sim.node,      // num nodes
                      default_sim.method,    // sim id
                      param(p, 0),           // amb temp
                      param(p, 3),           // uv intensity
                      param(p, 4),           // uv exposure time
                      default_sim.method,    // time stepping scheme
                      save_voxel,            // save voxel values
                      file_path,
                      mthread);
            sim.computeParticles(param(p, 1), param(p, 2));
            sim.simulate();

            #pragma omp critical
            {
                int thread_id = omp_get_thread_num();
                // std::cout << "Thread " << thread_id << std::endl;
                if (!std::isnan(sim.getObjective())) {
                    param(p, 5) = sim.getObjective();
                } else {
                    param(p, 5) = 1000.;
                }
            }
        }

        sort_data(param);

        // track top and average performers
        top_performer.push_back(param(0, param.cols() - 1));
        avg_parent.push_back(param.col(param.cols() - 1).head(P).mean());
        avg_total.push_back(param.col(param.cols() - 1).mean());

        std::cout << "\nparam: \n" << param << std::endl;
        // update input samples
        if (g < G - 1) {
            // mate top performing samples
            for (int i = 0; i < P; i+=2) {
                // generate random numbers
                lam_1 = distribution(gen);
                lam_2 = distribution(gen);

                // skip parent rows and update child rows
                param.row(i + C)     = lam_1 * param.row(i) + (1-lam_1) * param.row(i+1);
                param.row(i + C + 1) = lam_2 * param.row(i) + (1-lam_2) * param.row(i+1);
                
                // reset obj
                param(i+C,   param.cols()-1) = 1000.0;
                param(i+C+1, param.cols()-1) = 1000.0;
            }

            // generate new pop for remaining rows
            for (int i = P + C; i < pop; ++i) {
                param(i, 0) = c.min_temp + (c.max_temp - c.min_temp) * distribution(gen);
                param(i, 1) = c.min_rp   + (c.max_rp   - c.min_rp)   * distribution(gen);
                param(i, 2) = c.min_vp   + (c.max_vp   - c.min_vp)   * distribution(gen);
                param(i, 3) = c.min_uvi  + (c.max_uvi  - c.min_uvi)  * distribution(gen);
                param(i, 4) = c.min_uvt  + (c.max_uvt  - c.min_uvt)  * distribution(gen); 
                param(i, 5) = 1000.0;
            }

        }

        std::cout << "\ntop performer: " << std::endl;
        for (int i = 0; i < top_performer.size(); ++i) {
            std::cout << top_performer[i] << ", ";
        }
        std::cout << std::endl;

        std::cout << "\navg_parent: " << std::endl;
        for (int i = 0; i < avg_parent.size(); ++i) {
            std::cout << avg_parent[i] << ", ";
        }
        std::cout << std::endl;

        std::cout << "\navg_total: " << std::endl;
        for (int i = 0; i < avg_total.size(); ++i) {
            std::cout << avg_total[i] << ", ";
        }
        std::cout << std::endl;
    }

    // save performance vector to file
    std::ofstream top_performer_file;
    top_performer_file.open(file_path + "/top_performer.txt");
    top_performer_file << "top_performer, avg_parent, avg_total" << std::endl;
    for (int i = 0; i < top_performer.size(); ++i) {
        top_performer_file << top_performer[i] << ", " << avg_parent[i] << ", " << avg_total[i] << std::endl;
    }
    top_performer_file.close();

    // save param matrix to file
    std::ofstream param_file;
    param_file.open(file_path + "/param.txt");
    param_file << "temp, rp, vp, uvi, uvt, obj" << std::endl;
    param_file << param << std::endl;
    param_file.close();


    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = (std::chrono::duration_cast<std::chrono::microseconds>(stop - start)).count() / 1e6;
    
    std::ofstream time_file; 
    time_file.open(file_path + "/time_info.txt");
    time_file << " --- Simulation time: " << duration / 60 << "min ---";
    time_file.close();

    std::cout << " --- Simulation time: " << duration / 60 << "min ---" << std::endl;
    std::cout << " --- ----------------------------- ---" << std::endl;

    

    return 0;
}

void sort_data(Eigen::MatrixXd& PARAM){
    // Custom comparator for sorting by the fourth column in descending order
    auto comparator = [& PARAM](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        return a(PARAM.cols()-1) < b(PARAM.cols()-1);
    };

    // Convert Eigen matrix to std::vector of Eigen::VectorXd
    std::vector<Eigen::VectorXd> rows;
    for (int i = 0; i < PARAM.rows(); ++i) {
        rows.push_back(PARAM.row(i));
    }

    // Sort using the custom comparator
    std::sort(rows.begin(), rows.end(), comparator);

    // Copy sorted rows back to Eigen matrix
    for (int i = 0; i < PARAM.rows(); ++i) {
        PARAM.row(i) = rows[i];
    }
}
