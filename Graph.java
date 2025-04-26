import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Collectors;
import java.awt.geom.Rectangle2D;
import java.util.stream.Stream;

class Vertex {
    double x, y;
    int rank;
    int id;
    List<Edge> edges;

    public Vertex(int id, double x, double y) {
        this.id = id;
        this.x = x;
        this.y = y;

        this.edges = new ArrayList<>();

    }

    public List<Vertex> getConnectedVerticesStream() {
        return this.edges.stream()
                .map(e -> e.start == this ? e.end : e.start)
                .distinct()  // 自动去重
                .collect(Collectors.toList());
    }
    public boolean isDirectlyConnectedToAny(List<Vertex> vertices) {
        return edges.stream()
                .anyMatch(e -> vertices.contains(e.start) || vertices.contains(e.end));
    }
    // 从顶点列表中随机选择一个顶点并返回。
    public static Vertex getRandomVertex(List<Vertex> vertices) {
        if (vertices == null || vertices.isEmpty()) {
            throw new IllegalArgumentException("Vertex list cannot be null or empty");
        }
        Random random = new Random();
        int index = random.nextInt(vertices.size()); // 生成一个介于[0, vertices.size())的随机索引
        return vertices.get(index); // 返回随机索引对应的顶点
    }

    public double distanceTo(Vertex other) {
        if (other == null) return Double.POSITIVE_INFINITY;
        return Math.hypot(this.x - other.x, this.y - other.y);
    }



}


class Edge {

    Vertex start, end;
    double length;
    int v; // 车容量
    int n; // 当前车辆数
    int Level;
    public Edge(Vertex start, Vertex end, int v) {
        this.start = start;
        this.end = end;
        this.length = start.distanceTo(end);
        this.Level=0;
        this.v = v;
        this.n = 0; // 初始时道路上的车辆数为0
    }

    // 判断边是否连接给定的两个顶点
    public boolean connects(Vertex v1, Vertex v2) {
        return (start == v1 && end == v2) || (start == v2 && end == v1);
    }

    // Edge类新增方法：判断是否连接相同聚类
    public boolean linksSameClusters(Cluster a, Cluster b) {
        return (this.start.equals(a.getCore()) && this.end.equals(b.getCore())) ||
                (this.start.equals(b.getCore()) && this.end.equals(a.getCore()));
    }

    // 计算路段的通行时间，使用给定的公式
    public double getTrafficTime() {
        double f = (n <= v) ? 1 : (1 + Math.exp((double)n / v));
        return length * f;
    }

    // 更新车辆数
    public void updateTraffic(int cars) {
        this.n += cars;
       // System.out.println("Edge between " + start.id + " and " + end.id + " has " + n + " cars.");
    }
}

class UnionFind {
    int[] parent;
    int[] rank;

    public UnionFind(int n) {
        parent = new int[n];
        rank = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    public int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    public void union(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        if (rootX != rootY) {
            if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
        }
    }
}
class Cluster extends Graph {

    public List<Vertex> points;
    public int displayLevel;
    private Vertex medoid; // Medoid 中心点
    // 构造函数（递归构建层次结构）
    public Cluster(List<Vertex> points, int level) {
        this.points = new ArrayList<>(points);
        this.displayLevel = level;

    }

    // 计算 Medoid（簇内到其他点距离之和最小的点）
    public void calculateMedoid() {
        this.medoid = points.parallelStream()
                .min(Comparator.comparingDouble(v ->
                        points.stream()
                                .mapToDouble(u -> v.distanceTo(u))
                                .sum()
                ))
                .orElse(null);
    }

    public  Vertex getCore()
    {

        calculateMedoid();
        return medoid;
    }



}
// 分区类定义
class Partition {
    int rank; // 级别
    double Size;
    private final  Map<partKey, List<Vertex>> vertexPartMap; // 网格映射表
    private final Map<partKey, List<Edge>> edgePartMap ;
    // 网格坐标的键类（用于哈希表）
    private static class partKey {
        final int partX;
        final int partY;

        partKey(int partX, int partY) {
            this.partX = partX;
            this.partY = partY;
        }
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            partKey partKey = (partKey) o;
            return partX == partKey.partX&& partY == partKey.partY;
        }

        @Override
        public int hashCode() {
            return Objects.hash(partX, partY);
        }
    }
    public Partition(int rank,double size) {
        this.rank = rank;
        this.vertexPartMap=new HashMap<>();
        this.edgePartMap=new HashMap<>();
        this.Size=size;
    }

    // 将顶点分配到对应的网格
    public void addVertex(Vertex vertex) {
        int partX = (int) Math.floor(vertex.x / Size);
        int partY = (int) Math.floor(vertex.y / Size);
        partKey key = new partKey(partX, partY);
        vertexPartMap.computeIfAbsent(key, k -> new ArrayList<>()).add(vertex);
    }

    // 获取某个顶点所在网格及其邻近8个网格中的顶点
    public List<Vertex> getNearbyVertices(Vertex vertex) {
        List<Vertex> nearbyVertices = new ArrayList<>();
        int centerGridX = (int) Math.floor(vertex.x / Size);
        int centerGridY = (int) Math.floor(vertex.y / Size);

        // 遍历3x3的邻近网格区域
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                partKey key = new partKey(centerGridX + dx, centerGridY + dy);
                if ( vertexPartMap.containsKey(key)) {
                    nearbyVertices.addAll( vertexPartMap.get(key));
                }
            }
        }
        return nearbyVertices;
    }
    // 方法：将边插入网格索引
    public void addEdge(Edge edge) {
        double minX = Math.min(edge.start.x, edge.end.x);
        double maxX = Math.max(edge.start.x, edge.end.x);
        double minY = Math.min(edge.start.y, edge.end.y);
        double maxY = Math.max(edge.start.y, edge.end.y);

        int startPartX = (int) Math.floor(minX / Size);
        int endPartX = (int) Math.floor(maxX / Size);
        int startPartY = (int) Math.floor(minY / Size);
        int endPartY = (int) Math.floor(maxY / Size);

        // 将边注册到所有覆盖的网格中
        for (int x = startPartX; x <= endPartX; x++) {
            for (int y = startPartY; y <= endPartY; y++) {
                partKey key = new partKey(x, y);
                edgePartMap.computeIfAbsent(key, k -> new ArrayList<>()).add(edge);
            }
        }
    }
    public  boolean hasIntersection(Edge newEdge) {
        // 1. 获取新边覆盖的网格范围
        double minX = Math.min(newEdge.start.x, newEdge.end.x);
        double maxX = Math.max(newEdge.start.x, newEdge.end.x);
        double minY = Math.min(newEdge.start.y, newEdge.end.y);
        double maxY = Math.max(newEdge.start.y, newEdge.end.y);

        int startPartX = (int) Math.floor(minX / Size);
        int endPartX = (int) Math.floor(maxX / Size);
        int startPartY = (int) Math.floor(minY / Size);
        int endPartY = (int) Math.floor(maxY /Size);

        // 2. 遍历所有覆盖的网格，检查与现有边是否相交
        for (int x = startPartX; x <= endPartX; x++) {
            for (int y = startPartY; y <= endPartY; y++) {
                partKey key = new partKey(x, y);
                List<Edge> edgesInPart = edgePartMap.getOrDefault(key, Collections.emptyList());
                for (Edge existingEdge : edgesInPart) {
                    if (edgesIntersect(existingEdge, newEdge)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    // 线段相交判断工具函数
    private boolean edgesIntersect(Edge e1, Edge e2) {
        return linesIntersect(
                e1.start.x, e1.start.y, e1.end.x, e1.end.y,
                e2.start.x, e2.start.y, e2.end.x, e2.end.y
        );
    }

    // 几何工具函数（使用线段交叉判断）
    private boolean linesIntersect(double x1, double y1, double x2, double y2,
                                   double x3, double y3, double x4, double y4) {
        double d1 = direction(x3, y3, x4, y4, x1, y1);
        double d2 = direction(x3, y3, x4, y4, x2, y2);
        double d3 = direction(x1, y1, x2, y2, x3, y3);
        double d4 = direction(x1, y1, x2, y2, x4, y4);

        return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
                ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
    }

    private double direction(double xi, double yi, double xj, double yj, double xk, double yk) {
        return (xk - xi) * (yj - yi) - (xj - xi) * (yk - yi);
    }


}


class Graph {
    Partition parts;
    List<Vertex> vertices;
    Map<Integer, List<Edge>> rankedEdges = new HashMap<>();
    public Graph() {

        vertices = new ArrayList<>();
    }


    public List<Vertex> findNearestVertices(double x, double y, int count) {


        return vertices.stream()
                .sorted(Comparator.comparingDouble(v -> Math.sqrt(Math.pow(v.x - x, 2) + Math.pow(v.y - y, 2))))
                .limit(count)
                .collect(Collectors.toList());
    }

    public List<Edge> getRelatedEdges(List<Vertex> selectedVertices) {
        Set<Vertex> vertexSet = new HashSet<>(selectedVertices);
        return selectedVertices.stream()
                .flatMap(v -> v.edges.stream())
                .filter(e -> vertexSet.contains(e.start) && vertexSet.contains(e.end))
                .collect(Collectors.toList());
    }


    public void generateConnectedGraph(int N, double maxEdgeLength, double connectProbability, double maxCoordinateValue) {
        parts = new Partition(0, maxEdgeLength);
        long duration = data.measureTime(() -> {
            Random rand = new Random();
            for (int i = 0; i < N; i++) {
                double x = rand.nextDouble() * maxCoordinateValue * 1.5;
                double y = rand.nextDouble() * maxCoordinateValue;
                Vertex vertex = new Vertex(i, x, y);
                parts.addVertex(vertex);
                vertices.add(vertex);
            }
        });
        System.out.println("随机点生成耗时: " + duration + " ns");


        List<Edge> allEdges = new ArrayList<>();
         duration = data.measureTime(() -> {
             for (Vertex vertex : vertices) {
                 for (Vertex nearver : parts.getNearbyVertices(vertex))
                 // 遍历当前网格及8个邻近网格
                 {
                     if (vertex.id < nearver.id)// 避免重复
                     {
                         double distance = vertex.distanceTo(nearver);

                         if (distance <= maxEdgeLength) {
                             Edge edge = new Edge(vertex, nearver, (int) (10 + Math.random() * 10));
                             allEdges.add(edge);
                         }
                     }
                 }
             }
        });
        System.out.println(allEdges.size()+"条线段生成耗时: " + duration + " ns");


        duration = data.measureTime(() -> {
            allEdges.sort(Comparator.comparingDouble(e -> e.length));
        });
        System.out.println("排序线段耗时: " + duration + " ns");



        UnionFind uf = new UnionFind(vertices.size());

        duration = data.measureTime(() -> {
            for (Edge edge : allEdges) {
                int u = vertices.indexOf(edge.start);
                int v = vertices.indexOf(edge.end);

                if (uf.find(u) != uf.find(v) && !parts.hasIntersection(edge)) {
                    uf.union(u, v);
                    edge.start.edges.add(edge);
                    edge.end.edges.add(edge);
                    parts.addEdge(edge);
                }
            }
        });
        System.out.println("MST生成耗时: " + duration + " ns");



        // 添加额外边（避免交叉）
        duration = data.measureTime(() -> {


            for (Vertex vertex : vertices) {
                for (Vertex nearver : parts.getNearbyVertices(vertex))
                // 遍历当前网格及8个邻近网格
                {
                    if (vertex.id < nearver.id)// 避免重复
                    {
                        double distance = vertex.distanceTo(nearver);
                        if (distance <= maxEdgeLength && Math.random() < connectProbability) {
                            Edge edge = new Edge(vertex, nearver, (int) (10 + Math.random() * 10));
                            if (!parts.hasIntersection(edge)) {
                                parts.addEdge(edge);
                                edge.start.edges.add(edge);
                                edge.end.edges.add(edge);
                            }
                        }
                    }
                }
            }
        });
        System.out.println("添加随机边耗时: " + duration + " ns");

        duration = data.measureTime(() -> {

            int minpts=3;
            double eps=15;
            generateRank_edge(eps,minpts);
        });
        System.out.println("rank edge生成耗时: " + duration + " ns");


    }


    // 函数：根据两个顶点返回对应的边
    public Edge findEdge(Vertex v1, Vertex v2) {
        if (v1 == null || v2 == null) {
            return null;
        }

        // 遍历第一个顶点的边集合
        for (Edge edge : v1.edges) {
            // 如果边连接了这两个顶点，则返回这个边
            if (edge.connects(v1, v2)) {
                return edge;
            }
        }
        // 如果没有找到对应的边，则返回空
        return null;
    }

    public List<Vertex> getVertices() {
        return vertices;
    }

    public List<Edge> getEdges() {
        return vertices.stream()
                .flatMap(v -> v.edges.stream())
                .collect(Collectors.toList());
    }


    public void generateRank_edge(double eps, int minPts) {
        rankedEdges = new ConcurrentHashMap<>();
        rankedEdges.put(0, new CopyOnWriteArrayList<>(getEdges())); // 原始边在层级0
        List<Vertex> currentLevelPoints = vertices; // 初始为原始顶点集合
        double firstLevelEps = eps * Math.pow(2, 0); // level=1时指数为0
        List<Cluster> level1Clusters = clusterPoints(currentLevelPoints, firstLevelEps, minPts);
        List<Edge> previousEdges = rankedEdges.get(0);
        // 为每层计算中心点并生成下一级聚类
        for (int level = 1; level <= 3; level++) {

            // 1. 调整聚类参数（eps随层级指数增长）
            double currentEps = eps * Math.pow(2, level);

            // 2.1 分析拓扑时传入上一级边
            Map<Cluster, Set<Cluster>> topology = analyzeTopology(level1Clusters, previousEdges);

            // 2.2 生成抽象边（继承或合并上层边）
            int finalLevel = level;
            rankedEdges.put(level, new ArrayList<>());// 原始边在层级0
            topology.forEach((sourceCluster, connectedClusters) -> {
                connectedClusters.forEach(targetCluster -> {
                        Edge abstractEdge = createAbstractEdge(sourceCluster, targetCluster);
                        abstractEdge.Level= finalLevel;
                        abstractEdge.start.edges.add(abstractEdge);
                        abstractEdge.end.edges.add(abstractEdge);
                        // 分配边到当前层级（避免重复）
                        if (rankedEdges.get(finalLevel).stream().noneMatch(e -> e.equals(abstractEdge))) {
                            rankedEdges.get(finalLevel).add(abstractEdge);
                        }
                    });
                });

            // 3. 对中心点进行聚类
            currentLevelPoints=level1Clusters .stream()
                    .map(Cluster::getCore)
                    .filter(Objects::nonNull)       // 过滤无效中心点
                    .collect(Collectors.toList());
            level1Clusters = clusterPoints(currentLevelPoints, currentEps, minPts);
            previousEdges = rankedEdges.get(level);
        }

    }




    public List<Cluster> clusterPoints(List<Vertex> vertices, double eps, int  minPts) {

        Partition spatialIndex = new Partition(1,eps);
        vertices.forEach(spatialIndex::addVertex); // 非并行避免线程问题
        // 线程安全的数据结构
        Set<Vertex> visited = Collections.synchronizedSet(new HashSet<>());
        List<List<Vertex>> clusters = Collections.synchronizedList(new ArrayList<>());
        vertices.parallelStream().forEach(spatialIndex::addVertex);
        vertices.parallelStream().forEach(point -> {
            if (!visited.contains(point)) {
                visited.add(point);
                List<Vertex> neighbors = spatialIndex.getNearbyVertices(point);

                if (neighbors.size() >= minPts) {
                    List<Vertex> cluster = Collections.synchronizedList(new ArrayList<>());
                    cluster.add(point);
                    expandCluster(spatialIndex,point, neighbors, cluster, visited, minPts);
                    clusters.add(cluster);
                }
            }
        });

        //生成聚类结果

        return clusters.stream()
                .filter(clusterPoints -> !clusterPoints.isEmpty()) // 过滤空聚类
                .map(clusterPoints -> new Cluster(clusterPoints,0))
                .collect(Collectors.toList());
    }

    // 并行化集群扩展
    private void expandCluster(Partition spatialIndex ,
                               Vertex point,
                               List<Vertex> neighbors,
                               List<Vertex> cluster,
                               Set<Vertex> visited,
                               int minPts) {
        List<Vertex> frontier = new CopyOnWriteArrayList<>(neighbors);

        while (!frontier.isEmpty()) {
            List<Vertex> currentLevel = new ArrayList<>(frontier);
            frontier.clear();

            currentLevel.parallelStream().forEach(currentPoint -> {
                if (!visited.contains(currentPoint)) {
                    visited.add(currentPoint);
                    List<Vertex> newNeighbors = spatialIndex.getNearbyVertices(point);
                    if (newNeighbors.size() >= minPts) {
                        frontier.addAll(newNeighbors);
                    }
                }
                if (currentPoint != null && currentPoint.isDirectlyConnectedToAny(cluster)  && !cluster.contains(currentPoint))
               {
                   synchronized (cluster)  {
                            cluster.add(currentPoint);

                    }
                }


            });
        }
    }


    private Edge createAbstractEdge(Cluster a, Cluster b) {
        // 1. 获取两个聚类的核心点
        Vertex coreA = a.getCore();
        Vertex coreB = b.getCore();

        // 2. 检查核心点是否有效
        if (coreA == null || coreB == null) {
            // 处理无效核心点（例如返回空或抛出异常）
            throw new IllegalArgumentException("Cluster core vertex cannot be null");
        }

        // 3. 查找已存在的边
        Edge existingEdge = findEdge(coreA, coreB);

        // 4. 如果存在直接返回，否则创建新边
        return (existingEdge != null) ? existingEdge : new Edge(coreA, coreB, 0);
    }

    // 连接关系分析
    public Map<Cluster, Set<Cluster>> analyzeTopology(
            List<Cluster> currentClusters,
            List<Edge> previousEdges
    ) {
        Map<Vertex, Cluster> vertexClusterMap = new ConcurrentHashMap<>();

        // 构建顶点到簇的映射
        currentClusters.parallelStream().forEach(cluster ->
                cluster.points.parallelStream()
                        .forEach(p -> vertexClusterMap.put(p, cluster))
        );

        // 并行分析边连接
        Map<Cluster, Set<Cluster>> connections = new ConcurrentHashMap<>();
        previousEdges.parallelStream().forEach(edge -> {
            Cluster startCluster = vertexClusterMap.get(edge.start);
            Cluster endCluster = vertexClusterMap.get(edge.end);

            if (startCluster != null && endCluster != null && startCluster != endCluster) {
                connections.computeIfAbsent(startCluster, k -> ConcurrentHashMap.newKeySet())
                        .add(endCluster);
                connections.computeIfAbsent(endCluster, k -> ConcurrentHashMap.newKeySet())
                        .add(startCluster);
            }
        });

        return connections;
    }


    // 根据中心点查找对应的聚类
    private Cluster findClusterByMedoid(Vertex medoid, List<Cluster> clusters) {
        return clusters.parallelStream()
                .filter(c -> c.getCore().equals(medoid))
                .findFirst()
                .orElse(null);
    }

    public List<Vertex> calculateBestPath(Vertex source, Vertex destination) {
        Map<Vertex, Double> distances = new HashMap<>();
        Map<Vertex, Vertex> predecessors = new HashMap<>();
        PriorityQueue<Vertex> priorityQueue = new PriorityQueue<>(Comparator.comparingDouble(distances::get));

        vertices.forEach(vertex -> distances.put(vertex, Double.POSITIVE_INFINITY));
        distances.put(source, 0.0);
        priorityQueue.add(source);

        while (!priorityQueue.isEmpty()) {
            Vertex current = priorityQueue.poll();

            if (current == destination) {
                break;
            }

            current.edges.forEach(edge -> {
                Vertex neighbor = (edge.start == current) ? edge.end : edge.start;
                double newDist = distances.get(current) + edge.getTrafficTime();  // 使用通行时间作为权重

                if (newDist < distances.get(neighbor)) {
                    distances.put(neighbor, newDist);
                    predecessors.put(neighbor, current);
                    priorityQueue.add(neighbor);
                }
            });
        }

        // 构建路径，并确保路径上的顶点之间都有边
        List<Vertex> path = new ArrayList<>();
        for (Vertex at = destination; at != null; at = predecessors.get(at)) {
            path.add(at);
        }
        Collections.reverse(path);

        // 检查路径是否有效（每对相邻顶点之间都有边）
        for (int i = 0; i < path.size() - 1; i++) {
            Vertex v1 = path.get(i);
            Vertex v2 = path.get(i + 1);
            if (findEdge(v1, v2) == null) {
                return new ArrayList<>();  // 如果路径无效，返回空列表
            }
        }

        return path;
    }



    public List<Vertex> calculateShortestPath(Vertex source, Vertex destination) {
        Map<Vertex, Double> distances = new HashMap<>();
        Map<Vertex, Vertex> predecessors  = new HashMap<>();
        PriorityQueue<Vertex> priorityQueue = new PriorityQueue<>(Comparator.comparingDouble(distances::get));

        vertices.forEach(vertex -> distances.put(vertex, Double.POSITIVE_INFINITY));
        distances.put(source, 0.0);
        priorityQueue.add(source);

        while (!priorityQueue.isEmpty()) {
            Vertex current = priorityQueue.poll();

            if (current == destination) {
                break;
            }

            current.edges.forEach(edge -> {
                Vertex neighbor = (edge.start == current) ? edge.end : edge.start;
                double newDist = distances.get(current)+ edge.length;

                if (newDist < distances.get(neighbor)) {
                    distances.put(neighbor, newDist);
                    predecessors.put(neighbor, current);
                    priorityQueue.add(neighbor);
                }
            });
        }

        // 构建路径，并确保路径上的顶点之间都有边
        List<Vertex> path = new ArrayList<>();
        for (Vertex at = destination; at != null; at = predecessors.get(at)) {
            path.add(at);
        }
        Collections.reverse(path);

        // 检查路径是否有效（每对相邻顶点之间都有边）
        for (int i = 0; i < path.size() - 1; i++) {
            Vertex v1 = path.get(i);
            Vertex v2 = path.get(i + 1);
            if (findEdge(v1, v2) == null) {
                return new ArrayList<>();  // 如果路径无效，返回空列表
            }
        }

        return path;
    }

}
