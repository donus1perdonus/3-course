#include "visualizer.h"
#include "random_variable.h"
#include <iostream>
#include <limits>
#include <windows.h>
#include <set>

namespace {
	std::vector<std::pair<double, double>> readDistributionFromStdin() {
		std::vector<std::pair<double, double>> dist;
		std::cout << "Введите количество исходов n: ";
		int n = 0;
		while (!(std::cin >> n) || n <= 0) {
			std::cin.clear();
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			std::cout << "Некорректное n. Повторите: ";
		}

		std::set<double> usedValues;
		dist.reserve(static_cast<size_t>(n));
		for (int i = 0; i < n; ++i) {
			double x = 0.0, p = 0.0;
			std::cout << "Введите пару x p для i=" << (i + 1) << ": ";
			while (!(std::cin >> x >> p) || p < 0.0) {
				std::cin.clear();
				std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				std::cout << "Некорректный ввод. Ожидается: <x: число> <p: неотрицательное>. Повторите: ";
			}
			if (usedValues.count(x)) {
				std::cout << "Значение x=" << x << " уже встречалось. Введите уникальное значение.\n";
				--i;
				continue;
			}
			usedValues.insert(x);
			dist.emplace_back(x, p);
		}

		double sum = 0.0;
		for (const auto &pr : dist) sum += pr.second;
		const double eps = 1e-9;
		if (sum <= eps) {
			throw std::runtime_error("Сумма вероятностей равна нулю.");
		}
		if (std::abs(sum - 1.0) > 1e-6) {
			std::cout << "Сумма вероятностей = " << sum << ", будет выполнена нормализация к 1.\n";
			for (auto &pr : dist) pr.second /= sum;
		}
		return dist;
	}
}

int main(int argc, char** argv) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
	Visualizer visualizer;

	// Выбор источника распределения: CLI или пример по умолчанию
	bool useCli = false;
	for (int i = 1; i < argc; ++i) {
		std::string arg(argv[i]);
		if (arg == "--cli") useCli = true;
	}

	DiscreteRandomVariable rv1;
	if (useCli) {
		try {
			auto dist = readDistributionFromStdin();
			rv1.setDistribution(dist);
		} catch (const std::exception& ex) {
			std::cerr << "Ошибка ввода распределения: " << ex.what() << "\n";
			return -1;
		}
	} else {
		std::vector<std::pair<double, double>> dist1 = {
			{1, 0.1}, {2, 0.2}, {3, 0.4}, {4, 0.2}, {5, 0.1}
		};
		rv1.setDistribution(dist1);
	}

	std::cout << "Random Variable:" << std::endl;
	std::cout << rv1.toString() << std::endl;

	// Дополнительные операции-примеры
	DiscreteRandomVariable rv2 = rv1 * 2.0;
	std::cout << "X * 2:" << std::endl;
	std::cout << rv2.toString() << std::endl;

	DiscreteRandomVariable rv3 = rv1 + rv1;
	std::cout << "X + X:" << std::endl;
	std::cout << rv3.toString() << std::endl;

	// Инициализация окна после консольного ввода
	if (!visualizer.initialize(800, 600, "Discrete Random Variable Visualizer")) {
		return -1;
	}

	// Устанавливаем указатель на визуализатор для callback'ов
	glfwSetWindowUserPointer(visualizer.getWindow(), &visualizer);

	// Установка для визуализации
	visualizer.setRandomVariable(rv1);

	std::cout << "\nControls:" << std::endl;
	std::cout << "SPACE - Change view mode" << std::endl;
	std::cout << "ESC - Exit" << std::endl;

	visualizer.run();

	return 0;
}