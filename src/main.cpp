#include "visualizer.h"
#include "random_variable.h"
#include <iostream>
#include <limits>
#include <windows.h>
#include <set>
#include <fstream>

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

	void demonstrateTask1_1() {
		std::cout << "=== Задание 1.1: Демонстрация функционала DiscreteRandomVariable ===\n\n";

		// Выбор источника распределения: CLI или пример по умолчанию
		bool useCli = false;
		std::cout << "Использовать ввод с клавиатуры? (y/n): ";
		char choice;
		std::cin >> choice;
		if (choice == 'y' || choice == 'Y') {
			useCli = true;
		}

		DiscreteRandomVariable rv1;
		if (useCli) {
			try {
				auto dist = readDistributionFromStdin();
				rv1.setDistribution(dist);
			} catch (const std::exception& ex) {
				std::cerr << "Ошибка ввода распределения: " << ex.what() << "\n";
				return;
			}
		} else {
			std::vector<std::pair<double, double>> dist1 = {
				{1, 0.1}, {2, 0.2}, {3, 0.4}, {4, 0.2}, {5, 0.1}
			};
			rv1.setDistribution(dist1);
			std::cout << "Используется распределение по умолчанию: {1:0.1, 2:0.2, 3:0.4, 4:0.2, 5:0.1}\n\n";
		}

		// 1. Отображение в виде закона распределения
		std::cout << "1. Закон распределения:\n";
		std::cout << rv1.toString() << "\n";

		// 2. Отображение в виде полилайна
		std::cout << "2. Отображение в виде полилайна:\n";
		std::cout << rv1.toPolylineString() << "\n";

		// 3. Отображение в виде функции распределения
		std::cout << "3. Отображение в виде функции распределения:\n";
		std::cout << rv1.toCDFString() << "\n";

		// 4. Умножение на скаляр
		std::cout << "4. Умножение ДСВ на скаляр (X * 2.0):\n";
		DiscreteRandomVariable rv2 = rv1 * 2.0;
		std::cout << rv2.toString() << "\n";

		// 5. Сложение ДСВ
		std::cout << "5. Сложение ДСВ (X + X):\n";
		DiscreteRandomVariable rv3 = rv1 + rv1;
		std::cout << rv3.toString() << "\n";

		// 6. Умножение ДСВ
		std::cout << "6. Умножение ДСВ (X * X):\n";
		DiscreteRandomVariable rv4 = rv1 * rv1;
		std::cout << rv4.toString() << "\n";

		// 7. Статистические характеристики
		std::cout << "7. Статистические характеристики исходной ДСВ:\n";
		std::cout << "   Математическое ожидание: " << rv1.expectation() << "\n";
		std::cout << "   Дисперсия: " << rv1.variance() << "\n";
		std::cout << "   Среднеквадратическое отклонение: " << rv1.standardDeviation() << "\n";
		std::cout << "   Коэффициент асимметрии: " << rv1.skewness() << "\n";
		std::cout << "   Коэффициент эксцесса: " << rv1.kurtosis() << "\n\n";

		// 8. Сериализация и десериализация
		std::cout << "8. Сериализация и десериализация:\n";
		const std::string filename = "rv_serialized.bin";
		try {
			// Сериализация
			std::ofstream outFile(filename, std::ios::binary);
			if (!outFile.is_open()) {
				throw std::runtime_error("Не удалось открыть файл для записи");
			}
			rv1.serialize(outFile);
			outFile.close();
			std::cout << "   ДСВ сериализована в файл: " << filename << "\n";

			// Десериализация
			std::ifstream inFile(filename, std::ios::binary);
			if (!inFile.is_open()) {
				throw std::runtime_error("Не удалось открыть файл для чтения");
			}
			DiscreteRandomVariable rv5;
			rv5.deserialize(inFile);
			inFile.close();
			std::cout << "   ДСВ десериализована из файла: " << filename << "\n";
			std::cout << "   Проверка (должно совпадать с исходной):\n";
			std::cout << rv5.toString() << "\n";
		} catch (const std::exception& ex) {
			std::cerr << "   Ошибка при сериализации/десериализации: " << ex.what() << "\n";
		}

		std::cout << "\n=== Задание 1.1 завершено ===\n";
	}

	void demonstrateTask1_2() {
		std::cout << "=== Задание 1.2: Визуализация ДСВ через OpenGL ===\n\n";

		Visualizer visualizer;

		// Выбор источника распределения: CLI или пример по умолчанию
		bool useCli = false;
		std::cout << "Использовать ввод с клавиатуры? (y/n): ";
		char choice;
		std::cin >> choice;
		if (choice == 'y' || choice == 'Y') {
			useCli = true;
		}

		DiscreteRandomVariable rv1;
		if (useCli) {
			try {
				auto dist = readDistributionFromStdin();
				rv1.setDistribution(dist);
			} catch (const std::exception& ex) {
				std::cerr << "Ошибка ввода распределения: " << ex.what() << "\n";
				return;
			}
		} else {
			std::vector<std::pair<double, double>> dist1 = {
				{1, 0.1}, {2, 0.2}, {3, 0.4}, {4, 0.2}, {5, 0.1}
			};
			rv1.setDistribution(dist1);
			std::cout << "Используется распределение по умолчанию: {1:0.1, 2:0.2, 3:0.4, 4:0.2, 5:0.1}\n\n";
		}

		// Инициализация окна
		if (!visualizer.initialize(800, 600, "Discrete Random Variable Visualizer")) {
			std::cerr << "Ошибка инициализации визуализатора\n";
			return;
		}

		// Устанавливаем указатель на визуализатор для callback'ов
		glfwSetWindowUserPointer(visualizer.getWindow(), &visualizer);

		// Установка для визуализации
		visualizer.setRandomVariable(rv1);

		std::cout << "Controls:\n";
		std::cout << "SPACE - Change view mode\n";
		std::cout << "ESC - Exit\n\n";

		visualizer.run();

		std::cout << "\n=== Задание 1.2 завершено ===\n";
	}
}

int main(int argc, char** argv) {
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);

	// Парсинг аргументов командной строки
	if (argc < 2) {
		std::cerr << "Использование: " << argv[0] << " <номер_задания> [опции]\n";
		std::cerr << "Примеры:\n";
		std::cerr << "  " << argv[0] << " 1.1  - Демонстрация функционала DiscreteRandomVariable\n";
		std::cerr << "  " << argv[0] << " 1.2  - Визуализация ДСВ через OpenGL\n";
		return -1;
	}

	std::string taskNumber = argv[1];

	if (taskNumber == "1.1") {
		demonstrateTask1_1();
	} else if (taskNumber == "1.2") {
		demonstrateTask1_2();
	} else {
		std::cerr << "Неизвестный номер задания: " << taskNumber << "\n";
		std::cerr << "Доступные задания: 1.1, 1.2\n";
		return -1;
	}

	return 0;
}