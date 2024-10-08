#include <bits/stdc++.h>
using namespace std;
// We don't need to write the functions for dfs and bfs in a graph
// Two things to remember: we can only use the adjancency matrix or adjancency list representation of the graph
// For dfs use stack, for bfs use queue (or use a dequeue and be cautious about the popping)
// Tip for visited set, use a vector<bool> to represent the state of nodes whether they are visited or not.
// If we need to remember the route just use a stack path 
// Using bfs we can talk about the rank of a node from a source (how many steps), while with dfs we can talk about the depth of a node from a source (how many steps) and we can also get the min or max depth...

// --------------! Find the shortest path !----------------
// Dijkstra algorithm on an adjancency list
const int INF = 1000000000;
vector<vector<pair<int, int>>> adj;

// d is a vector containing the distances, p is a vector containing the ancestor (to get the path), which means we can delete p if we need to
// Ofc we can modify the algorithm to suit our needs (check this problem: https://codeforces.com/problemset/problem/2014/E)
// We should use a heap if we want to make the algorithm faster
void dijkstra(int s, vector<int> & dist, vector<int> & p) {
    int n = adj.size();
    dist.assign(n, INF); //distances
    p.assign(n, -1); //traceback
    vector<bool> u(n, false);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, s});
    dist[s] = 0;
    while(!pq.empty()){
        pair<int, int> front = pq.top(); pq.pop();
        int d = front.first, u = front.second;
        if(d > dist[u]) continue;
        for(pair<int, int> e: adj[u]){
            if(dist[u] + e.second < dist[e.first]){
                dist[e.first] = dist[u] + e.second;
                p[e.first] = u;
                pq.push({dist[e.first], e.first});
            }
        }
    }
}
vector<int> restore_path(int s, int t, vector<int> const& p) {
    vector<int> path;

    for (int v = t; v != s; v = p[v])
        path.push_back(v);
    path.push_back(s);

    reverse(path.begin(), path.end());
    return path;
}

// If the path are negative dijkstra doesn't work here and we need to use bellman ford algorithm
struct Edge {
    int a, b, cost;
};

int n, m, v;
// it's more convenient to use a list of edges
vector<Edge> edges;
const int INF = 1000000000;

void bellleman_ford()
{
    vector<int> d(n, INF);
    d[v] = 0; // v is the source vertes
    for (int i = 0; i < n - 1; ++i)
        for (Edge e : edges)
            if (d[e.a] < INF)
                d[e.b] = min(d[e.b], d[e.a] + e.cost);
    // display d, for example, on the screen
}

// What if we need to find the shortest from all pairs of (source, dest) possible, then we need to use Floyd-Warshall algorithm
// d is a 2d array (adjancency matrix)
for (int k = 0; k < n; ++k) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            d[i][j] = min(d[i][j], d[i][k] + d[k][j]); 
        }
    }
}

// --------------! Find connected components !----------------
// G is an undirected graph with n nodes and m edges,
// We are required to find in it all the connected components, i.e, several groups of vertices such that within a group each vertex can be reached from another and no path exists between different groups.
// Complexity: O(E + V)
 int n;
vector<vector<int>> adj;
vector<bool> used;
vector<int> comp;
vector<vector<int>> connected_groups;

void dfs(int v) {
    used[v] = true ;
    comp.push_back(v);
    for (int u : adj[v]) {
        if (!used[u])
            dfs(u);
    }
}

void find_comps() {
    fill(used.begin(), used.end(), 0);
    for (int v = 0; v < n; ++v) {
        if (!used[v]) {
            comp.clear(); // Comp stores the connected components of a group one at time after the dfs
            dfs(v);
            connected_groups.push_back(comp); // We store the connected components
        }
    }
}

// What if we want to find bridges in a graph, that means finding all the edges that, if they were to be deleted, the graph will become disconnected (remembers me of problem A from the 1st contest)
// This algorith runs in O(V + E) time
void IS_BRIDGE(int v,int to); // some function to process the found bridge when we find the bridge we do something.
int n; // number of nodes
vector<vector<int>> adj; // adjacency list of graph

vector<bool> visited;
vector<int> tin, low;
int timer;

void dfs(int v, int p = -1) {
    visited[v] = true;
    tin[v] = low[v] = timer++;
    bool parent_skipped = false;
    for (int to : adj[v]) {
        if (to == p && !parent_skipped) {
            parent_skipped = true;
            continue;
        }
        if (visited[to]) {
            low[v] = min(low[v], tin[to]);
        } else {
            dfs(to, v);
            low[v] = min(low[v], low[to]);
            if (low[to] > tin[v])
                IS_BRIDGE(v, to);
        }
    }
}

void find_bridges() {
    timer = 0;
    visited.assign(n, false);
    tin.assign(n, -1);
    low.assign(n, -1);
    for (int i = 0; i < n; ++i) {
        if (!visited[i])
            dfs(i);
    }
}

// What is instead of searching for an edge that if deleted makes the graph disconnected, we want to find the vertices that if they get deleted along with the associated edges, make the graph disconnected
void IS_CUTPOINT(vertex); // Some function that let us process them
int n; // number of nodes
vector<vector<int>> adj; // adjacency list of graph

vector<bool> visited;
vector<int> tin, low;
int timer;

void dfs(int v, int p = -1) {
    visited[v] = true;
    tin[v] = low[v] = timer++;
    int children=0;
    for (int to : adj[v]) {
        if (to == p) continue;
        if (visited[to]) {
            low[v] = min(low[v], tin[to]);
        } else {
            dfs(to, v);
            low[v] = min(low[v], low[to]);
            if (low[to] >= tin[v] && p!=-1)
                IS_CUTPOINT(v);
            ++children;
        }
    }
    if(p == -1 && children > 1)
        IS_CUTPOINT(v);
}

void find_cutpoints() {
    timer = 0;
    visited.assign(n, false);
    tin.assign(n, -1);
    low.assign(n, -1);
    for (int i = 0; i < n; ++i) {
        if (!visited[i])
            dfs (i);
    }
}

// What if we want to find SCC, which means a subset of vertices C such that:
// for all u, v, in C if u != v then there exist a path from u to v and from v to u
// C is maximal, which means if a vertex is added to C the first condition will be lost
// We use Kosaraju's algorithm

// --------------! Spanning Trees !----------------
// I guess we all kknow what is a spanning tree, so let's just dive in with the algorithms.
// If we have an adjancency list or adjancency matrix, just go for primm's algorithm
// This functions returns an adjancency list of the MST
struct Edge {
    int w = INF, to = -1;
};
int n;
vector<vector<Edge>> adj; // adjacency matrix of graph
const int INF = 1000000000; // weight INF means there is no edge

class Compare{

    bool operator()(pair<int, Edge> a, pair<int, Edge> b){return a.second.w > b.second.w; }
};

vector<vector<Edge>> MST;
void prim(vector<vector<Edge>>& MST) {
    int total_weight = 0;
    vector<bool> selected(n, false); // We suppose that the vertices are 0 indexed
    int num_of_selected = 0;
    priority_queue<pair<int, Edge>, vector<pair<int, Edge>>, Compare> pq;
    selected[0] = true;
    num_of_selected++;
    for(Edge e : adj[0]){
        pq.push({0, e});
    }
    while(!pq.empty() || num_of_selected < adj.size()){
        pair<int, Edge> e = pq.top(); pq.pop();
        if(selected[e.second.to]) continue;
        MST[e.first].push_back(e.second);
        Edge to_add; to_add.w = e.second.w; to_add.to = e.first;
        MST[e.second.to].push_back(to_add);
        selected[e.second.to] = true;
        num_of_selected++;
        for(Edge f : adj[e.second.to]){
            pq.push({e.second.to, f});
        }
    }
}

// ------------! Eulerian Path !----------------
//A Eulerian path is a path in a graph that passes through all of its edges exactly once. A Eulerian cycle is a Eulerian path that is a cycle.
//The problem is to find the Eulerian path in an undirected multigraph with loops.
// Theorem, an eulerian path exists <=> the degree of all the vertices is even
// And an Eulerian path exists if and only if the number of vertices with odd degrees is two
// The program below searches for and outputs a Eulerian loop or path in a graph, or outputs -1 if it does not exist.
// Notice that we use an adjacency matrix in this problem
int n, m;
vector<vector<pair<int, int>>> g;
vector<int> path;
vector<bool> seen;

void dfs(int node) {
	while (!g[node].empty()) {
		auto [son, idx] = g[node].back();
		g[node].pop_back();
		if (seen[idx]) { continue; }
		seen[idx] = true;
		dfs(son);
	}
	path.push_back(node);
}
// In main we get the input, we see if the number of edges in all nodes is even and then run dfs(0) to fill the path with the nodes
// If there's exaclty 2 vertices with odd degree (exactly 2) start dfs from 1 of them, the answer should be contain a path starting from the latter and ending at the other vertex

// --------------! Lowest common ancestor !-----
// Preprocess the graph from a node using dfs and we get 2 arrays height and euler
// {euler[i], height[i]} being the height of the visited node euler[i] during time i (number of steps from source)
// Then if they want the LCA of two nodes it's like RMQ from indices 
// first[i] is the first appearance of the node i in the traversal
struct LCA {
    vector<int> height, euler, first, segtree;
    vector<bool> visited;
    int n;

    LCA(vector<vector<int>> &adj, int root = 0) {
        n = adj.size();
        height.resize(n);
        first.resize(n);
        euler.reserve(n * 2);
        visited.assign(n, false);
        dfs(adj, root);
        int m = euler.size();
        segtree.resize(m * 4);
        build(1, 0, m - 1);
    }

    void dfs(vector<vector<int>> &adj, int node, int h = 0) {
        visited[node] = true;
        height[node] = h;
        first[node] = euler.size();
        euler.push_back(node);
        for (auto to : adj[node]) {
            if (!visited[to]) {
                dfs(adj, to, h + 1);
                euler.push_back(node);
            }
        }
    }

    void build(int node, int b, int e) /*node = 1st node or source of dfs (1), b = 0, e = euler.size() - 1 */ {
        if (b == e) {
            segtree[node] = euler[b];
        } else {
            int mid = (b + e) / 2;
            build(node << 1, b, mid);
            build(node << 1 | 1, mid + 1, e);
            int l = segtree[node << 1], r = segtree[node << 1 | 1];
            segtree[node] = (height[l] < height[r]) ? l : r;
        }
    }

    int query(int node, int b, int e, int L, int R) {
        if (b > R || e < L)
            return -1;
        if (b >= L && e <= R)
            return segtree[node];
        int mid = (b + e) >> 1;

        int left = query(node << 1, b, mid, L, R);
        int right = query(node << 1 | 1, mid + 1, e, L, R);
        if (left == -1) return right;
        if (right == -1) return left;
        return height[left] < height[right] ? left : right;
    }

    int lca(int u, int v) {
        int left = first[u], right = first[v];
        if (left > right)
            swap(left, right);
        return query(1, 0, euler.size() - 1, left, right);
    }
};
