# DS683 - Homework Assignment 2

## Learning Goals
I tested two core properties of GNNs:

Node-Level Equivariance: If we shuffle the nodes in the input, the output features should move to the exact same new positions. Tested on the Cora dataset.


<img width="945" height="47" alt="Screenshot 2026-03-18 at 8 56 36 PM" src="https://github.com/user-attachments/assets/6623fec5-cd30-48fd-b4b0-63429bb9dfb2" />


Graph-Level Invariance: The overall graph embedding should stay exactly the same regardless of node order, provided we use a symmetric readout like sum, mean, or max pooling. Tested on the MUTAG dataset.


<img width="945" height="471" alt="Screenshot 2026-03-18 at 9 05 27 PM" src="https://github.com/user-attachments/assets/f4518973-f936-4409-9a2a-6107506f0f0d" />


How to run the tests
python3 tests.py



The Counterexample
As part of the assignment, I included a "bad" readout—specifically, taking the embedding of node 0.

In a GNN, global layers like `global_add_pool`, `global_mean_pool`, and `global_max_pool` are mathematically symmetric; they ensure the graph embedding remains identical whether we sum all node features, average them, or take the maximum value across the group. However, simply taking the embedding of "node 0" fails this test. Because node 0 is just a fixed index in the data matrix, shuffling the graph puts a completely different node in that first slot, causing the resulting graph-level representation to change entirely. This confirms that while message-passing captures local structure, only symmetric pooling can produce a stable, permutation-invariant representation for the whole graph.

<img width="945" height="425" alt="Screenshot 2026-03-18 at 9 23 13 PM" src="https://github.com/user-attachments/assets/97b2a78b-e952-48e9-950a-d3ccc7f653ef" />
