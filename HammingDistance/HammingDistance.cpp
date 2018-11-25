#include "pch.h"
#include <vector>
#include <iterator>
#include <string>
#include <iostream>
#include <random>
#include <chrono>

using namespace std;

long hammingDistance(std::string w1, std::string w2)
{
	if (w1.length() != w2.length())
		throw;
	long distance = 0;
	for (long i = 0; i < w1.length(); i++)
		if (w1[i] != w2[i])
			distance++;
	return distance;
}
	
void gen_random(string &s, const long len)
{
	static const char alphanum[] =
		"0123456789"
		"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		"abcdefghijklmnopqrstuvwxyz";
	s = "";
	for (long i = 0; i < len; ++i) {
		s += alphanum[rand() % (sizeof(alphanum) - 1)];
	}
}

long main()
{
	string word1;
	string word2;
	long length = 1;
	length = 31474836;// 314748364;
	gen_random(word1, length);
	word2 = word1;
	auto start = std::chrono::high_resolution_clock::now();
	long distance = hammingDistance(word1, word2);
	auto finish = std::chrono::high_resolution_clock::now();
	auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
	std::cout << "Length: " << length << " " <<  microseconds.count() << " microseconds\n";
	//std::cout
	//	<< "Hamming distance of worlds:\n"
	//	//<< word1 
	//	<< "\n"
	//	//<< word2 
	//	<< "\nis: "
	//	<< distance;
	
}