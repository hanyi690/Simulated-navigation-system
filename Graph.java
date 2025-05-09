
import javafx.util.Pair;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;


class Vertex {
    double x, y;
    int rank;
    int id;
    List<Edge>[] edges;  // 按Level分组的邻接边（数组存储）

    public Vertex(int id, double x, double y) {
        this.id = id;
        this.x = x;
        this.y = y;
        // 初始化4个层级的边列表（假设层级为0-3）
        this.edges = new ArrayList[4];
        for (int i = 0; i < 4; i++) {
            edges[i] = new ArrayList<>();
        }

    }

    public List<Vertex> getConnectedVerticesStream() {

        return this.edges[0].stream()
                .map(e -> e.start == this ? e.end : e.start)
                .distinct()  // 自动去重
                .collect(Collectors.toList());
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

    public double distanceToNoVertex(double x,double y) {
        return Math.hypot(this.x - x, this.y - y);
    }

}


class Edge {

    Vertex start, end;
    double length;
    int v; // 车容量
    int n; // 当前车辆数
    List<Edge> abstractEdge;
    public Edge(Vertex start, Vertex end, int v) {
        this.start = start;
        this.end = end;
        this.length = start.distanceTo(end);
        this.abstractEdge=new ArrayList<>();
        this.v = v;
        this.n = 0; // 初始时道路上的车辆数为0
    }

    // 判断边是否连接给定的两个顶点
    public boolean connects(Vertex v1, Vertex v2) {
        return (start == v1 && end == v2) || (start == v2 && end == v1);
    }

    public  Vertex otherVertex(Vertex v)
    {
     return  this.start==v?this.end:this.start;
    }
    // 计算路段的通行时间，使用给定的公式
    public double getTrafficTime() {
        double f = (n <= v) ? 1 : (1 + Math.exp((double)n / v));
        return length * f;
    }

    // 更新车辆数
    public void updateTraffic(int cars) {
        this.n += cars;
        for(Edge e:abstractEdge)
        {
            e.updateTraffic(cars);
        }
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

class PointGroup implements Comparable<PointGroup>{
        List<Vertex> points;      // 合并后的点集
         Vertex medoid;
         int id;//同层唯一区分
        //Set<Edge> connectedEdges; // 保留的外部连接
         int zoomLevel;            // 适用缩放层级
    // 在构造函数中识别边界边
    public PointGroup(List<Vertex> points ,int level,int i) {
                 this.zoomLevel=level;
                 this.points = new ArrayList<>(points);
                 this.id=i;
           /*      this.connectedEdges = points.stream().flatMap(v -> v.edges[level].stream())
                         .filter(e -> !points.contains(e.otherVertex(v)))
                         .collect(Collectors.toSet());*/
    }
    // 实现Comparable接口，按id比较
    @Override
    public int compareTo(PointGroup other) {
        return Integer.compare(this.id, other.id);
    }
    // 计算 Medoid（簇内到其他点距离之和最小的点）
     public void calculateMedoid() {
         if (points.size() == 1) {
             // 单点组直接返回该点作为质心
             this.medoid = points.get(0);
         } else {
             // 多点组计算最小总距离的质心
             this.medoid = points.parallelStream()
                     .min(Comparator.comparingDouble(v ->
                             points.stream()
                                     .mapToDouble(v::distanceTo)
                                     .sum()
                     ))
                     .orElse(null); // 此处理论上不会触发，因为 points 至少有一个点
         }

    }
    public  Vertex getCore()     {     calculateMedoid();return medoid;     }
}
// 分区类定义
class Partition {
    double Size;
    protected final  Map<partKey, List<Vertex>> vertexPartMap; // 网格映射表
    private final Map<partKey, List<Edge>> edgePartMap ;
    // 网格坐标的键类（用于哈希表）
    protected static class partKey {
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
    public Partition(double size) {
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
                        return false;
                    }
                }
            }
        }
        return true;
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


    public List<Vertex> findNearestVertices(double x, double y, int count,Partition parts) {
        // 参数校验
        if (count <= 0) {
            return Collections.emptyList();
        }

        // 使用最大堆维护最近的count个点
        PriorityQueue<Vertex> nearestVertices = new PriorityQueue<>(
                (v1, v2) -> Double.compare(v2.distanceToNoVertex(x, y), v1.distanceToNoVertex(x, y))
        );

        int centerGridX = (int) Math.floor(x / parts.Size);
        int centerGridY = (int) Math.floor(y / parts.Size);

        int radius = 0;
        boolean foundEnough = false;
        boolean shouldStop = false;

        while (!shouldStop) {

            boolean currentRoundHasPoints = false;

            // 搜索当前半径的外围环
            for (int dx = -radius; dx <= radius; dx++) {
                for (int dy = -radius; dy <= radius; dy++) {
                    // 只搜索当前半径的外围环
                    if (Math.abs(dx) != radius && Math.abs(dy) != radius) {
                        continue;
                    }

                    Partition.partKey key = new Partition.partKey(centerGridX + dx, centerGridY + dy);
                    if (parts.vertexPartMap.containsKey(key)) {
                        currentRoundHasPoints = true;
                        for (Vertex vertex : parts.vertexPartMap.get(key)) {
                            nearestVertices.offer(vertex);
                            if (nearestVertices.size() > count) {
                                nearestVertices.poll(); // 保持堆大小不超过count
                            }
                        }
                    }
                }
            }
            radius++;
            // 检查终止条件
            if (nearestVertices.size() >= count) {
                if (foundEnough) {
                    // 已经完成额外一圈搜索，可以停止
                    shouldStop = true;
                } else {
                    // 首次找到足够点数，标记并继续搜索下一圈
                    foundEnough = true;

                }
            } else if (!currentRoundHasPoints  ) {
                // 当前圈没有找到点且已达到最大半径，停止搜索
                shouldStop = true;
            }
        }

        // 转换为List并按距离升序排列
        return new ArrayList<>(nearestVertices);
    }

    public List<Edge> getRelatedEdges(List<Vertex> selectedVertices) {
        Set<Vertex> vertexSet = new HashSet<>(selectedVertices);
        return selectedVertices.stream()
                .flatMap(v -> v.edges[0].stream())
                .filter(e -> vertexSet.contains(e.start) && vertexSet.contains(e.end))
                .collect(Collectors.toList());
    }


    public void generateConnectedGraph(int N, double maxEdgeLength, double connectProbability, double maxCoordinateValue) {
        parts = new Partition( maxEdgeLength);
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


        duration = data.measureTime(() -> allEdges.sort(Comparator.comparingDouble(e -> e.length)));
        System.out.println("排序线段耗时: " + duration + " ns");



        UnionFind uf = new UnionFind(vertices.size());

        duration = data.measureTime(() -> {
            for (Edge edge : allEdges) {
                int u = vertices.indexOf(edge.start);
                int v = vertices.indexOf(edge.end);

                if (uf.find(u) != uf.find(v) && parts.hasIntersection(edge)) {
                    uf.union(u, v);
                    edge.start.edges[0].add(edge);
                    edge.end.edges[0].add(edge);
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
                            if (parts.hasIntersection(edge)) {
                                parts.addEdge(edge);
                                edge.start.edges[0].add(edge);
                                edge.end.edges[0].add(edge);
                            }
                        }
                    }
                }
            }
        });
        System.out.println("添加随机边耗时: " + duration + " ns");

        duration = data.measureTime(() -> {

            int minpts=5;
            double eps=15;
            generateRankedEdges(eps,minpts);
        });
        System.out.println("rank edge生成耗时: " + duration + " ns");


    }


    // 函数：根据两个顶点返回对应的边
    public Edge findEdge(Vertex v1, Vertex v2,int level) {
        if (v1 == null || v2 == null) {
            return null;
        }

        // 遍历第一个顶点的边集合
        for (Edge edge : v1.edges[level]) {
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
                .flatMap(v -> v.edges[0].stream())
                .distinct()
                .collect(Collectors.toList());
    }


    public void generateRankedEdges(double eps, int minPts) {
        rankedEdges = new ConcurrentHashMap<>();
        List<Edge> originalEdges = getEdges();
        rankedEdges.put(0, new CopyOnWriteArrayList<>(originalEdges)); // 初始层级

        List<Vertex> currentVertices = new CopyOnWriteArrayList<>(vertices);

        // 层级生成主循环
        for (int level = 0; level < 3; level++) { // 0~3共4个层级
            double currentEps  =  eps * Math.pow(2, level);

            // 1. 执行点合并
            ConcurrentHashMap<Vertex, PointGroup> vertexToGroupMap = new ConcurrentHashMap<>();
            List<PointGroup> groups = mergePoints(currentVertices,level,currentEps,vertexToGroupMap);
            // 所有顶点必须被映射到组
            assert currentVertices.stream().allMatch(v -> vertexToGroupMap.containsKey(v));

            int count=groups.parallelStream()
                    .filter(Objects::nonNull)
                    .mapToInt(group ->
                            (group.points != null) ? group.points.size() : 0
                    )
                    .sum();
            System.out.println((level+1)+"层有"+groups.size()+"个点，涵盖上层"+count+ "个点");
            // 2. 生成抽象边（跳过第0层）
            System.out.println("rankedge"+(level)+" "+rankedEdges.get(level).size());

            buildAbstractEdges( groups,rankedEdges.get(level), level+1,vertexToGroupMap);
            Set<Vertex> uniqueVertices = ConcurrentHashMap.newKeySet();

            rankedEdges.get(level+1).parallelStream().forEach(edge -> {
                uniqueVertices.add(edge.start);
                uniqueVertices.add(edge.end);
            });
            System.out.println("rankedge"+(level+1)+"包括点数"+uniqueVertices.size() );
            // 3. 准备下一层输入（使用质心）
            currentVertices.clear();
            currentVertices = groups.stream()
                    .map(PointGroup::getCore)  // 直接使用缓存的质心
                    .filter(Objects::nonNull)  // 过滤无效核心点
                    .collect(Collectors.toCollection(CopyOnWriteArrayList::new));
            System.out.println("输入下一层的点数"+ currentVertices.size());
            groups.clear();
        }
    }




    public List<PointGroup> mergePoints(List<Vertex> vs, int zoomLevel,double mergeDistance,ConcurrentHashMap<Vertex, PointGroup> vertexToGroupMap) {
        // 阶段1：构建空间索引
        Partition index = new Partition(mergeDistance);
        vs.forEach(index::addVertex);

        // 阶段2：合并邻近连通点
        Set<Vertex> processed = ConcurrentHashMap.newKeySet();
        List<PointGroup> groups = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger i= new AtomicInteger();
        vs.parallelStream().forEach(v -> {
            // 原子性检查并标记处理
            if (processed.add(v)) { // 直接使用 add(v) 的返回值

                PointGroup currentGroup = new PointGroup(new ArrayList<>(), zoomLevel, i.getAndIncrement());
                currentGroup.points.add(v);
                vertexToGroupMap.put(v, currentGroup);

                // 处理邻近点（同样需原子性操作）
                index.getNearbyVertices(v).stream()
                        .filter(k -> isConnected(v, k, mergeDistance, zoomLevel))
                        .filter(k -> processed.add(k))
                        .forEach(k -> {
                            currentGroup.points.add(k);
                            vertexToGroupMap.put(k, currentGroup);
                        });
                    groups.add(currentGroup);
            }
        });
        // 阶段3：处理未访问的点
        vs.parallelStream().forEach(v -> {
            // 原子性检查并标记处理
            if (processed.add(v)) {
                PointGroup singleGroup = new PointGroup(Collections.singletonList(v), zoomLevel, i.getAndIncrement());
                groups.add(singleGroup);
                vertexToGroupMap.put(v, singleGroup);
            }
        });

        return groups;
    }

    void buildAbstractEdges(List<PointGroup> groups,List<Edge> originalEdges,int level,ConcurrentHashMap<Vertex, PointGroup> vertexToGroupMap) {
        // 步骤1：识别组间连接边

        Map<Pair<PointGroup, PointGroup>, Set<Edge>> groupConnections = new HashMap<>();
        for (Edge e : originalEdges) {
            PointGroup g1 = vertexToGroupMap.get(e.start);
            PointGroup g2 = vertexToGroupMap.get(e.end);
            Vertex A=g1.getCore();
            Vertex B=g2.getCore();
            if (A != null && B != null &&A.id!= B.id ){
                //System.out.println("生成");
                PointGroup minGroup = (A.id<B.id) ? g1 : g2;
                PointGroup maxGroup = (A.id<B.id) ? g2 : g1;
                Pair<PointGroup, PointGroup> key = new Pair<>(minGroup, maxGroup);
                groupConnections.computeIfAbsent(key, k -> new HashSet<>()).add(e);
            }
        }

        rankedEdges.put(level,new ArrayList<>());
        // 步骤2：生成抽象边
        for (Map.Entry<Pair<PointGroup, PointGroup>, Set<Edge>> entry : groupConnections.entrySet()) {
            Edge abstractEdge = createAbstractEdge(entry.getKey().getValue(),entry.getKey().getKey(),calculateCombinedWeight(entry.getValue()), level);

            // 维护映射关系
            rankedEdges.get(level).add(abstractEdge);
            abstractEdge.start.edges[level].add(abstractEdge);
            abstractEdge.end.edges[level].add(abstractEdge);
            for (Edge e : entry.getValue()) {
                e.abstractEdge.add(abstractEdge);
            }
        }
    }

    Edge createAbstractEdge(PointGroup a,PointGroup b,int v, int level){

        if (a == null || b == null) {
            throw new IllegalArgumentException("PointGroups cannot be null");
        }
        // 获取两个聚类的核心点
        Vertex coreA = a.getCore();
        Vertex coreB = b.getCore();

        // 检查核心点是否有效
            if (coreA == null || coreB == null) {
                // 处理无效核心点（例如返回空或抛出异常）
                throw new IllegalArgumentException("Cluster core vertex cannot be null");
            }
            // 3. 查找已存在的边
            Edge existingEdge = findEdge(coreA, coreB,level);

            // 4. 如果存在直接返回，否则创建新边
            return (existingEdge != null) ? existingEdge : new Edge(coreA, coreB, v);

    }


    // 辅助方法：计算抽象边权重
    int calculateCombinedWeight(Set<Edge> edges) {
        return edges.stream().mapToInt(e -> e.v).sum();
    }

    private boolean isConnected(Vertex a, Vertex b,double mergeDistance, int level) {
        // 条件1：空间距离足够近
        boolean isNear = a.distanceTo(b) <= mergeDistance;
        // 条件2：存在直接连接或路径连通
        boolean hasDirectLink = a.edges[level].stream()
                .anyMatch(e -> e.connects(a, b));

        //boolean hasPathLink = findshortestPath(a, b, zoomLevel) <mergeDistance;

        return isNear && (hasDirectLink );// //hasPathLink);
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

            current.edges[0].forEach(edge -> {

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
            if (findEdge(v1, v2,0) == null) {
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

            current.edges[0].forEach(edge -> {
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
            if (findEdge(v1, v2,0) == null) {
                return new ArrayList<>();  // 如果路径无效，返回空列表
            }
        }

        return path;
    }

}
