# Heuristic State Space Search: a Light Introduction to Python

This repository contains a lecture (`slides.pdf`) introducing Python and a partial implementation of a grid-world planner (`<main|lib>.py'`) to introduce state space search and Python debugging. The lecture is interactive, and involves students following along in the Python REPL and then editing and running short scripts, followed by a quick introduction to graph search algorithms (requiring whiteboard examples). It is recommended to take a short stretch break between the Python material and the search material (about 1 hour in). Once the search slides and whiteboard examples are done, students clone this repository, and work through issues relating to unmet dependencies and unimplemented features. The goal is to practice looking through someone else's unfamiliar code, and identifying the parts that are relevant to addressing the problem described in the error messages. The code includes complete implementations of a problem generator and a search-run visualizer, so that students can develop an intuitive sense for the behavior of different search algorithms experientially. Possible extensions include generating a suite of instances of different sizes, and plotting results (e.g., expansions vs. problem size, or vs. obstacle density) per algorithm, comparing the performance of different search orderings.

## Dependencies, Installation, and Usage

Installation of the code and its dependencies is covered in the slides (somewhat Socratically). Usage is well-documented by the code's command-line argument parsing; pass the `--help` flag to learn more.

## Attribution

The slides were drafted with ChatGPT, and edited and extended by hand. The Python code was written by hand.
