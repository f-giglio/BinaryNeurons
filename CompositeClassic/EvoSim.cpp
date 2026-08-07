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
#include <ncurses.h>

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

void draw(const std::bitset<N>& state)
{
    erase();
    mvprintw(0, 0, "Current State");

    constexpr int rows = 25;
    constexpr int cols = 40;

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {

            const int index = row * cols + col;

            const int y = row + 2;
            const int x = col * 2;

            if (state[index]) {
                attron(A_REVERSE);
                mvaddstr(y, x, "  ");
                attroff(A_REVERSE);
            }
            else {
                mvaddstr(y, x, "  ");
            }
        }
    }

    mvprintw(
        rows + 4,
        0,
        "SPACE -> Update    Q -> Exit    R -> Randomize"
    );

    refresh();
}

// -----------------------------------------------------------------------------
// MAIN
int main()
{
    double W = 0.1;
    double k = 100;
    double eta_w = 1.0;
    double eta_s = 1.0;
	int step = 0;
	bool fixed_point_found = false;
	bool cycle_found = false;

	int fixed_point_step = 0;
	int cycle_step = 0;
	int cycle_length = 0;

    std::bitset<N> State;
    Eigen::MatrixXd w;
    Eigen::MatrixXi w_strong;

    uint32_t seed = 15123455;

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> ran_u(0.0, 1.0);
    std::normal_distribution<double> ran_g(0.0, 1.0);

	std::unordered_map<std::bitset<N>, int> visited;

    init(
        N,
        W,
        k,
        eta_w,
        eta_s,
        State,
        w,
        w_strong,
        gen,
        ran_u,
        ran_g
    );

    randomize_state(State, ran_u, gen);

	visited.clear();
	visited[State] = 0;
	step = 0;

    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0);

    bool running = true;

    while (running) {

        draw(State);

    	mvprintw(
    	    29, 0,
    	    "N=%d  W=%.2f  k=%.0f  eta_w=%.2f  eta_s=%.2f  step=%d",
    	    N, W, k, eta_w, eta_s, step
    	);

    	if (fixed_point_found) {
    	    mvprintw(
    	        31, 0,
    	        "FIXED POINT found at step %d",
    	        fixed_point_step
    	    );
    	}

    	if (cycle_found) {
    	    mvprintw(
    	        31, 0,
    	        "CYCLE found at step %d - length %d",
    	        cycle_step,
    	        cycle_length
    	    );
    	}

    	refresh();

    	const int key = getch();

        switch (key) {

			case ' ': {
			    std::bitset<N> previous = State;
			
			    update_state(State, w);
			    step++;
			
			    if (State == previous && fixed_point_found == false) {
			        fixed_point_found = true;
			        fixed_point_step = step;
			    }
			    else {
			        auto it = visited.find(State);
				
			        if (it != visited.end() && cycle_found == false) {
			            cycle_found = true;
			            cycle_step = step;
			            cycle_length = step - it->second;
			        }
			        else {
			            visited[State] = step;
			        }
			    }
			
			    break;
			}

			case 'r':
			case 'R':
			    randomize_state(State, ran_u, gen);

			    visited.clear();
			    visited[State] = 0;

			    step = 0;

			    fixed_point_found = false;
			    cycle_found = false;

			    fixed_point_step = 0;
			    cycle_step = 0;
			    cycle_length = 0;

			    break;

            case 'q':
            case 'Q':
                running = false;
                break;
        }
    }

    endwin();

    return 0;
}