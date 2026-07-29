#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <cstdint>
#include <limits>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <thread>
#include <set>
#include <bitset>
#include <unordered_set>
#include <array>
#include <algorithm>

#define N 1000
#define BestTop 100

using namespace Eigen;

// With synchronous update, performs an exploration.
// Weight matrix has a variable degree of symmetry in the strong and asymmetric weak connections.
// We keep fixed the value of neurons N.

// We keep fixed: N = 1000

// We explore for:
//   k       100
//   ETA_s = 0 -> 1
//   ETA_w = 0 -> 1
//   W       0 -> 5



void randomize_state(std::bitset<N>& state,
                     std::uniform_real_distribution<double>& ran_u,
                     std::mt19937& gen) {

	for (int i = 0; i < N; ++i) {
		state[i] = ran_u(gen) < 0.1;
	}
}



void init(const int n,
          const double W,
          const double k,
          const double eta_w,
          const double eta_s,
          std::bitset<N>& state,
          Eigen::MatrixXd& w,
          Eigen::MatrixXi& w_strong,
          std::mt19937& gen,
          std::uniform_real_distribution<double>& ran_u,
          std::normal_distribution<double>& ran_g) {

	double c = (double)k / (double)n;

	state.reset();

	w.resize(n, n);
	w.setZero();

	w_strong.resize(n, n);
	w_strong.setZero();

	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			double a = ran_g(gen);
			double b = ran_g(gen);
			w(i, j) = a;
			w(j, i) = eta_w * a + std::sqrt(1.0 - eta_w * eta_w) * b;
		}
	}

	w /= std::sqrt((double)n);

	const double p_symm = c * (c + eta_s * (1.0 - c));
	const double p_asymm   = c * (1.0 - c) * (1.0 - eta_s);

	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			double prob = ran_u(gen);

			if (prob < p_symm) {
				w_strong(i, j) = 1;
				w_strong(j, i) = 1;
			} else if (prob < p_symm + p_asymm) {
				w_strong(i, j) = 1;
			} else if (prob < p_symm + 2 * p_asymm) {
				w_strong(j, i) = 1;
			}
		}
	}

	w += W * w_strong.cast<double>();
	w.diagonal().setZero();
}



void update_state(std::bitset<N>& state,
                  const Eigen::MatrixXd& w) {

	Eigen::VectorXd h;
	h.resize(state.size());
	h.setZero();

	for (int i = 0; i < N; i++) {
		if (state[i] == 1) {
			h += w.col(i);
		}
	}

	std::array<int, N> indexes;
	std::iota(indexes.begin(), indexes.end(), 0);


	std::nth_element(
	    indexes.begin(),
	    indexes.begin() + BestTop,
	    indexes.end(),
	[&h](int a, int b) {
		if (h(a) != h(b)) {
			return h(a) > h(b);
		}
		return a < b;
	}
	);

	state.reset();

	for (int i = 0; i < BestTop; ++i) {
		if (h(indexes[i]) > 0) {
			state.set(indexes[i]);
		}
	}
}



//struct VecLess {
//	bool operator()(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const {
//		for (Eigen::Index i = 0; i < a.size(); ++i) {
//			if (a[i] < b[i]) return true;
//			if (b[i] < a[i]) return false;
//		}
//		return false;
//	}
//};



//void stability_count(const Eigen::MatrixXd& w,
//                     const std::set<Eigen::VectorXd, VecLess>& Fixed_Points,
//                     Eigen::MatrixXd& Stability,
//                     const bool normalize) {
//
//	Stability.resize(Fixed_Points.size(), w.rows());
//	Stability.setZero(Fixed_Points.size(), w.rows());
//
//	int count = 0;
//	for (const auto& point : Fixed_Points) {
//		Eigen::VectorXd h = w * point;
//		Eigen::VectorXd stab;
//		if (normalize) {
//			stab = (h.cwiseProduct(point)) / w.rows();
//	} else {
//			stab = h.cwiseProduct(point);
//		}
//		Stability.row(count) = stab.transpose();
//		count++;
//	}
//}



//void write_stability(const int n,
//                     const double W,
//                     const double k,
//                     const double eta_w,
//                     const double eta_s,
//                     const Eigen::MatrixXd& Stability) {
//
//	std::ostringstream filename;
//	filename << "Stability"
//	         << "_N" << std::fixed << std::setprecision(3) << n
//	         << "_W" << std::fixed << std::setprecision(3) << W
//	         << "_k" << std::fixed << std::setprecision(3) << k
//	         << "_etaw" << std::fixed << std::setprecision(3) << eta_w
//	         << "_etas" << std::fixed << std::setprecision(3) << eta_s
//	         << ".csv";
//
//	std::ofstream file(filename.str());
//	Eigen::IOFormat csv_format(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n");
//	file << Stability.format(csv_format);
//	file.close();
//}



void write_points(const std::unordered_set<std::bitset<N>>& Fixed_Points) {
	std::ofstream file("FixedPoints.csv");

	for (const auto& point : Fixed_Points) {
		for (int i = 0; i < N; ++i) {
			if (i > 0)
				file << ',';
			file << point[i];
		}
		file << '\n';
	}
}



std::unordered_set<std::bitset<N>> converge(const int n,
                                  const int cycles,
                                  const double W,
                                  const double k,
                                  const double eta_w,
                                  const double eta_s,
                                  const int max_steps,
                                  std::bitset<N>& state,
                                  Eigen::MatrixXd& w,
                                  Eigen::VectorXd& record,
                                  Eigen::MatrixXd& stability,
                                  std::uniform_real_distribution<double>& ran_u,
std::mt19937& gen) {

	Eigen::VectorXd num_updates;
	std::bitset<N> initial_state;

	double sum_hamming = 0.0, mean_updates = 0.0, mean_hamming = 0.0, mean_stability = 0.0;
	int fp_runs = 0, cycle_runs = 0, max_steps_run = 0;


	num_updates.resize(cycles);
	num_updates.setZero(cycles);


	std::unordered_set<std::bitset<N>> Fixed_Points;
	std::unordered_set<std::bitset<N>> Visited_Points;


	bool fp_found = false, cycle_found = false;


	for (int cycle = 0; cycle < cycles; cycle++) {
		fp_found = false;
		cycle_found = false;

		randomize_state(state, ran_u, gen);
		initial_state = state;

		Visited_Points.clear();
		Visited_Points.insert(state);

		int steps = 0;

		while (!fp_found && !cycle_found) {

			std::bitset<N> prev_state = state;

			update_state(state, w);

			if (prev_state == state) {
				fp_found = true;
			}

			if (!fp_found) {
				if (!Visited_Points.insert(state).second) {
					cycle_found = true;
				} else {
					steps++;
				}
			}
			if (steps >= max_steps && !fp_found && !cycle_found) {
				break;
			}
		}


		if (fp_found) {
			Fixed_Points.insert(state);
			fp_runs++;
			num_updates[cycle] = steps;

			int hamming = (initial_state ^ state).count();
			sum_hamming += hamming;
		}
		
		else if (cycle_found){
			cycle_runs++;
		}
		
		else{
			max_steps_run++;
		}
	}


#ifdef DEBUG

	//One could place here the debug checks, but for now none is useful to us

#endif

	if (fp_runs != 0) {
		mean_updates = (float)num_updates.sum() / (float)fp_runs;
		mean_hamming = (float)sum_hamming / (float)fp_runs;
	}
	std::cout << "------------------------------------\n" << std::endl;
	std::cout << "\nN: " << n << "\tW: " << W << "\tk: " << k << "\teta_w: " << eta_w << "\teta_s: " << eta_s << std::endl;
	std::cout << "\nFixed Points Runs: " << fp_runs << std::endl;
	std::cout << "\nCycles Runs: " << cycle_runs << std::endl;
	std::cout << "\nMax Step Runs: " << max_steps_run << std::endl;
	std::cout << "\nFixed Points found: " << Fixed_Points.size() << std::endl;
	std::cout << "\nMean of updates: " << mean_updates << std::endl;
	std::cout << "\nMean of hamming distance of convergence: " << mean_hamming << std::endl;


	//stability_count(w, Fixed_Points, stability, true);
	//write_stability(n, W, k, eta_w, eta_s, stability);


	//if (stability.size() > 0) {
	//    mean_stability = stability.rowwise().mean().sum();
	//}


	record(0) = n;
	record(1) = W;
	record(2) = k;
	record(3) = eta_w;
	record(4) = eta_s;
	record(5) = fp_runs;
	record(6) = cycle_runs;
	record(7) = max_steps_run;
	record(8) = Fixed_Points.size();
	record(9) = mean_updates;
	record(10) = mean_hamming;

	return Fixed_Points;
}



void write_weights(const int n,
                   const double W,
                   const double k,
                   const double eta_w,
                   const double eta_s,
                   const Eigen::MatrixXd& w) {

	std::ostringstream filename;
	filename << "Weights" << ".csv";

	std::ofstream file(filename.str());
	Eigen::IOFormat csv_format(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n");
	file << w.format(csv_format);
	file.close();
}



void write_strong(const int n,
                  const double W,
                  const double k,
                  const double eta_w,
                  const double eta_s,
                  const Eigen::MatrixXi& w_strong) {

	std::ostringstream filename;
	filename << "StrongWeights" << ".csv";

	std::ofstream file(filename.str());
	Eigen::IOFormat csv_format(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n");
	file << w_strong.format(csv_format);
	file.close();
}



void write_data(const Eigen::MatrixXd& data) {
	std::ofstream file("Data.csv");
	file << "N,W,k,ETA_w,ETA_s,fp_runs,cycles_runs,max_steps_runs,fixed_points,updates,hamming\n";
	Eigen::IOFormat csv_format(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n");
	file << data.transpose().format(csv_format);
	file.close();
}


// -----------------------------------------------------------------------------
// MAIN
int main() {
	omp_set_num_threads(24);

	int cycles = 100, max_steps = 500;

	double eta_min = 0, eta_max = 1, Incr_eta = 0.05;

	double W_min = 0, W_max = 0.5, Incr_W = 0.05;
	double k = 100;

	std::cout << "N: " << N << std::endl;
	std::cout << "W: " << W_min << " -> " << W_max << std::endl;
	std::cout << "k: " << k << std::endl;
	std::cout << "ETA_w: " << eta_min << " -> " << eta_max << std::endl;
	std::cout << "ETA_s: " << eta_min << " -> " << eta_max << std::endl;

	int steps_N = 1;
	int steps_W = 11;
	int steps_k = 1;
	int steps_eta_w = 21;
	int steps_eta_s = 21;
	int total_runs = (int)std::lround(steps_N * steps_W * steps_k * steps_eta_w * steps_eta_s);

	Eigen::MatrixXd Data;
	Data.setZero(11, total_runs);

	#pragma omp parallel for collapse(5) schedule(dynamic)
	for (int iN = 0; iN < steps_N; iN++) {
		for (int iW = 0; iW < steps_W; ++iW) {
			for (int ik = 0; ik < steps_k; ++ik) {
				for (int iew = 0; iew < steps_eta_w; ++iew) {
					for (int ies = 0; ies < steps_eta_s; ++ies) {

						std::bitset<N> State;
						Eigen::MatrixXd w;
						Eigen::MatrixXi w_strong;
						Eigen::VectorXd Record;
						Eigen::MatrixXd Stability;


						//int N = N_min + iN * Incr_N;
						double W = W_min + iW * Incr_W;
						//double k = k_min + ik * Incr_k;
						double eta_w = eta_min + iew * Incr_eta;
						double eta_s = eta_min + ies * Incr_eta;

						Record.setZero(11);


						uint32_t seed = 151234553u ^ (uint32_t)(iN * 1002003u + iW * 10000019u + ik * 10007u + iew * 101u + ies * 1009u);
						std::mt19937 gen(seed);
						std::uniform_real_distribution<double> ran_u(0.0, 1.0);
						std::normal_distribution<double> ran_g(0.0, 1.0);

						init(N, W, k, eta_w, eta_s, State, w, w_strong, gen, ran_u, ran_g);

						//write_weights(N, W, k, eta_w, eta_s, w);
						//write_strong(N, W, k, eta_w, eta_s, w_strong);

						std::unordered_set<std::bitset<N>> FP = converge(N, cycles, W, k, eta_w, eta_s, max_steps, State, w, Record, Stability, ran_u, gen);
						//write_points(FP);
						int index = ((((iN * steps_W + iW) * (steps_k) + ik) * steps_eta_w + iew) * steps_eta_s + ies);
						Data.col(index) = Record;

						std::cout << (index + 1) << "/" << total_runs << std::endl;
						std::cout << "------------------------------------\n" << std::endl;
					}
				}
			}
		}
	}

	write_data(Data);

	std::cout << "\n\n\nTest Ended\n------------------------------------------------------------------------------------------" << std::endl;
}
