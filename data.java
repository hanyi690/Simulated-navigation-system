import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.*;
import javafx.scene.input.ScrollEvent;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;


class Vertex {
    double x, y;

    int id;
    List<Edge> edges;

    public Vertex(int i,double x, double y) {
        this.id = i;
        this.x = x;
        this.y = y;
        this.edges = new ArrayList<>();
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
        return Math.sqrt(Math.pow(this.x - other.x, 2) + Math.pow(this.y - other.y, 2));
    }
}

class Edge {

    Vertex start, end;
    double length;
    int v; // 车容量
    int n; // 当前车辆数

    public Edge(Vertex start, Vertex end, int v) {
        this.start = start;
        this.end = end;
        this.length = start.distanceTo(end);
        this.v = v;
        this.n = 0; // 初始时道路上的车辆数为0
    }

    // 判断边是否连接给定的两个顶点
    public boolean connects(Vertex v1, Vertex v2) {
        return (start == v1 && end == v2) || (start == v2 && end == v1);
    }

    // 计算路段的通行时间，使用给定的公式，以分钟为单位
    public double getTrafficTime() {
        double f = (n <= v) ? 1 : (1 + Math.exp((double)n / v));
        return length * f;
    }

    // 更新车辆数
    public void updateTraffic(int cars) {
        this.n += cars;
        //System.out.println("Edge between " + start.id + " and " + end.id + " has " + n + " cars.");
    }
}

class Graph {
    List<Vertex> vertices;

    public Graph() {
        this.vertices = new ArrayList<>();
    }

    public Map<Integer, Vertex> generateRandomVertices(int N, double maxCoordinateValue) {
        Map<Integer, Vertex> pointsMap = new HashMap<>();
        Random rand = new Random();
        for (int i = 0; i < N; i++) {
            double x = rand.nextDouble() * maxCoordinateValue*1.5;
            double y = rand.nextDouble() * maxCoordinateValue;
            vertices.add(new Vertex(i,x, y));
            pointsMap.put(i, new Vertex(i,x,y));
        }
        return pointsMap;
    }

    public List<Vertex> findNearestVertices( Map<Integer, Vertex> pointsMap, int pointid,int count) {
        double x = pointsMap.get(pointid).x;
        double y = pointsMap.get(pointid).y;
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


    public void generateConnectedGraph(double maxEdgeLength, double connectProbability) {
        List<Edge> allEdges = new ArrayList<>();
        List<Edge> existingEdges = new ArrayList<>();

        for (int i = 0; i < vertices.size(); i++) {
            for (int j = i + 1; j < vertices.size(); j++) {
                Edge edge = new Edge(vertices.get(i), vertices.get(j), (int)(10 +Math.random() * 10));
                if (edge.length <= maxEdgeLength) {
                    allEdges.add(edge);
                }
            }
        }

        allEdges.sort(Comparator.comparingDouble(e -> e.length));

        UnionFind uf = new UnionFind(vertices.size());
        List<Edge> mstEdges = new ArrayList<>();

        for (Edge edge : allEdges) {
            int u = vertices.indexOf(edge.start);
            int v = vertices.indexOf(edge.end);

            if (uf.find(u) != uf.find(v) && !hasIntersection(existingEdges, edge)) {
                uf.union(u, v);
                mstEdges.add(edge);
                existingEdges.add(edge);
                edge.start.edges.add(edge);
                edge.end.edges.add(edge);
            }
        }

        // 添加额外边（避免交叉）
        for (int i = 0; i < vertices.size(); i++) {
            for (int j = i + 1; j < vertices.size(); j++) {
                double distance = vertices.get(i).distanceTo(vertices.get(j));
                if (distance <= maxEdgeLength && Math.random() < connectProbability) {
                    Edge edge = new Edge(vertices.get(i), vertices.get(j), (int)(10 + Math.random() * 10));
                    if (!hasIntersection(existingEdges, edge)) {
                        existingEdges.add(edge);
                        edge.start.edges.add(edge);
                        edge.end.edges.add(edge);
                    }
                }
            }
        }
    }
    private boolean hasIntersection(List<Edge> edges, Edge newEdge) {
        for (Edge edge : edges) {
            if (edgesIntersect(edge.start, edge.end, newEdge.start, newEdge.end)) {
                return true;
            }
        }
        return false;
    }

    private boolean edgesIntersect(Vertex a1, Vertex a2, Vertex b1, Vertex b2) {
        return linesIntersect(a1.x, a1.y, a2.x, a2.y, b1.x, b1.y, b2.x, b2.y);
    }

    // 几何工具函数（使用线段交叉判断）
    private boolean linesIntersect(double x1, double y1, double x2, double y2,
                                   double x3, double y3, double x4, double y4) {
        double d1 = direction(x3, y3, x4, y4, x1, y1);
        double d2 = direction(x3, y3, x4, y4, x2, y2);
        double d3 = direction(x1, y1, x2, y2, x3, y3);
        double d4 = direction(x1, y1, x2, y2, x4, y4);

        if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
                ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
            return true;
        }

        return false;
    }

    private double direction(double xi, double yi, double xj, double yj, double xk, double yk) {
        return (xk - xi) * (yj - yi) - (xj - xi) * (yk - yi);
    }

    // 函数：根据两个顶点返回对应的边
    public Edge findEdge(Vertex v1, Vertex v2) {
        if (v1 == null || v2 == null)
        {
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

    public void scaleMap(double scaleFactor) {
        vertices.forEach(vertex -> {
            vertex.x *= scaleFactor;
            vertex.y *= scaleFactor;
        });
    }

    public List<Vertex> calculateShortestPath(Vertex source, Vertex destination) {
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
}

class Car implements Comparable<Car> {
    private Vertex currentVertex; // 当前所在点
    private Vertex destinationVertex; // 前往点
    private double travelTime; // 到达下一个地点耗时t
    private double timer; // 计时器c
    private List<Vertex> path; // 车辆的路径
    private TimeSlot currentSlot; // 当前所属时间槽
    @Override
    public int compareTo(Car other) {
        return Double.compare(this.timer, other.timer);
    }

    public void setCurrentSlot(TimeSlot slot) { this.currentSlot = slot; }
    public TimeSlot getCurrentSlot() { return currentSlot; }
    public double getTimer() {
        return timer;
    }

    public Car(Vertex currentVertex,Graph graph) {
        this.currentVertex = currentVertex;
        this.destinationVertex = null;
        this.travelTime = 0;
        this.timer = 0;
        this.path = new ArrayList<>();
        setRandomDestination(currentVertex,graph);
    }


    // 更新车辆状态，包括计时器和路径
    public boolean update(double deltaTime, TrafficSimulation simulation) {
        Graph graph = simulation.getGraph();
        timer -= deltaTime;
        boolean if_change=false;
        if (timer <= 0) {
            if_change=true;
            if (path != null && !path.isEmpty()) {
                Vertex lastVertex = currentVertex;
                currentVertex = path.remove(0);

                if (!path.isEmpty()) {
                    Vertex nextVertex = path.get(0);
                    Edge lastEdge = graph.findEdge(lastVertex, currentVertex);
                    Edge currentEdge = graph.findEdge(currentVertex, nextVertex);

                    // TODO: 2025/4/8 lastEdge会为空
                    // 检查 lastEdge 和 currentEdge 是否为空
                    if (lastEdge != null) {
                        lastEdge.updateTraffic(-1);  // 离开上一条边
                    }
                    if (currentEdge != null) {
                        currentEdge.updateTraffic(+1);  // 进入下一条边
                        travelTime = currentEdge.getTrafficTime();
                        timer = travelTime;
                    } else {
                        // 如果没有边，直接跳到下一个顶点
                        timer = 1.0;  // 默认时间
                    }
                } else {
                    setRandomDestination(currentVertex, graph);
                }
            } else {
                setRandomDestination(currentVertex, graph);
            }
        }
        return  if_change;
    }

    // 设置随机目的地，并计算路径
    public void setRandomDestination(Vertex currentVertex, Graph graph) {
        List<Vertex> possibleDestinations = graph.getVertices();
        if (possibleDestinations.isEmpty()) return;

        int maxAttempts = 10;  // 最多尝试 10 次
        int attempts = 0;

        while (attempts < maxAttempts) {
            destinationVertex = Vertex.getRandomVertex(possibleDestinations);
            if (destinationVertex == currentVertex) continue;  // 不能选择自己作为目的地

            path = graph.calculateShortestPath(currentVertex, destinationVertex);
            if (path != null && !path.isEmpty()) {
                return;  // 找到有效路径
            }
            attempts++;
        }

        // 如果尝试多次仍然无效，保持当前状态（不更新 path）
        path = new ArrayList<>();
    }

}
//按时间槽（n * timestep 到 (n+1) * timestep）分组存储车辆
  class  TimeSlot  implements   Comparable<TimeSlot>{
    private final double startTime; // 时间槽起始时间（不可变）
    private final List<Car> cars = new ArrayList<>();

    public TimeSlot(double startTime) {
        this.startTime = startTime;
    }

    public double getStartTime() { return startTime; }
    public List<Car> getCars() { return cars; }

    @Override
    public int compareTo(TimeSlot other) {
        return Double.compare(this.startTime, other.startTime);
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

class  TrafficSimulation {

    private final PriorityQueue<TimeSlot> timeSlotsQueue ;
    private final Map<Double, TimeSlot> timeSlotsMap ; // 模拟中的车辆列表
    private Graph graph; // 路网图
    private double simulationTime; // 模拟的总时间（以分钟为单位）
    private double currentTime; // 当前模拟时间（以分钟为单位）
    private double timeStep; // 模拟时间步长（以分钟为单位）
    private double min_timer;//车队列中最小的timer
    private double deltatime;
    public TrafficSimulation(Graph graph, double simulationTime, double timeStep,int cars_num) {
        this.graph = graph;
        this.simulationTime = simulationTime;
        this.timeStep = timeStep;
        this.currentTime = 0;
        this.min_timer= Double.MAX_VALUE;
        this.deltatime=0;
        this.timeSlotsQueue = new PriorityQueue<>();
        this.timeSlotsMap= new HashMap<>();
        initializeCars( cars_num);
    }

    private void initializeCars(int cars_num) {
        // 初始化车辆，为每辆车设置起始点和目的地
        Random rand = new Random();
        for (int i = 0; i < cars_num; i++) { // 假设我们初始化20000辆车
            List<Vertex> possiblever=graph.getVertices();
            Vertex start = Vertex.getRandomVertex(possiblever);
            Car car=new Car(start, graph);
            // 计算车辆应属的时间槽起始时间
            double newStartTime = Math.floor(car.getTimer() / timeStep) * timeStep;

            // 获取或创建新时间槽
            TimeSlot newSlot = timeSlotsMap.get(newStartTime);
            if (newSlot == null) {
                newSlot = new TimeSlot(newStartTime);
                timeSlotsMap.put(newStartTime, newSlot);
                timeSlotsQueue.add(newSlot);
            }

            // 将车辆加入新时间槽
            newSlot.getCars().add(car);
            car.setCurrentSlot(newSlot);
        }
        min_timer=timeSlotsQueue.peek().getStartTime()+timeStep;
    }

    public void startSimulation() {
        new Thread(()->{
        while (currentTime < simulationTime) {
            updateSimulation();
            currentTime += timeStep;
             //模拟休眠一段时间来模拟现实时间流逝，休眠时间取决于时间步长和模拟速度
            try {
                Thread.sleep((long) (timeStep*10 ));
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        }).start();
    }

    public   Graph getGraph()
    {
        return  graph;
    }

    private void updateSimulation() {
        // 更新所有车辆的状态
        deltatime += timeStep;
        // 循环处理所有满足时间条件的车辆
        if (!timeSlotsQueue.isEmpty() && deltatime >= min_timer) {
            // 暂存需要重新调度的车辆
            List<Car> rescheduledCars = new ArrayList<>();
            while (!timeSlotsQueue.isEmpty() && min_timer <= deltatime) {
                // 从旧时间槽中移除车辆
                for (Car car:timeSlotsQueue.peek().getCars()) {
                    rescheduledCars.add(car);
                    car.update(deltatime, this);
                }
                // 如果旧时间槽为空，则清理
                timeSlotsMap.remove(timeSlotsQueue.peek().getStartTime());
                timeSlotsQueue.remove(timeSlotsQueue.peek());
            }
            for (TimeSlot timeslot : timeSlotsQueue) {
                for(Car car: timeslot.getCars()) {
                    car.update(deltatime, this); // 每帧持续更新，与定时器无关
                }
            }
            while (!rescheduledCars.isEmpty() ) {
                Car car=rescheduledCars.remove(0);
                double newStartTime = Math.floor(car.getTimer() / timeStep) * timeStep;
                // 获取或创建新时间槽
                TimeSlot newSlot = timeSlotsMap.get(newStartTime);
                if (newSlot == null) {
                    newSlot = new TimeSlot(newStartTime);
                    timeSlotsMap.put(newStartTime, newSlot);
                    timeSlotsQueue.add(newSlot);
                }
                // 将车辆加入新时间槽
                newSlot.getCars().add(car);
                car.setCurrentSlot(newSlot);
            }
            min_timer=timeSlotsQueue.peek().getStartTime()+timeStep;
          deltatime=0;
    }
}
}


public class data extends Application {

    private double scaleFactor = 1.0;
    private static final double SCALE_SPEED = 0.05;
    private static final double MIN_SCALE = 0.1;
    private static final double MAX_SCALE = 5.0;

    private double translateX = 0, translateY = 0;  // 用于平移地图的位置
    private double mousePressedX, mousePressedY;

    AtomicInteger judgeshortest = new AtomicInteger(1);

    private List<Vertex> shortestPath = new ArrayList<>();
    AtomicReference<List<Vertex>> nearestVertices = new AtomicReference<>(new ArrayList<>());;
    AtomicReference<List<Edge>> relatedEdges =new AtomicReference<>(new ArrayList<>());

    @Override
    public void start(Stage primaryStage) {

        int N = 5000;
        double maxCoordinateValue = 1000;
        double maxEdgeLength = 500;
        double connectProbability=0.2;

        Graph graph = new Graph();
        Map<Integer, Vertex> vertexMap =graph.generateRandomVertices(N, maxCoordinateValue);
        graph.generateConnectedGraph(maxEdgeLength,connectProbability);

        if (graph.getVertices().isEmpty()) {
            System.err.println("Error: Graph has no vertices.");
            return;
        }
        if (graph.getVertices().size() < 2) {
            System.err.println("Error: Not enough vertices in the graph.");
            return;
        }


        // 创建 Canvas 绘制图形
        // Group root = new Group();
        Canvas canvas = new Canvas(1800, 1000);
        // root.getChildren().add(canvas);

        GraphicsContext gc = canvas.getGraphicsContext2D();

        gc.setLineWidth(2);

        //如果 graph.getVertices() 为空或 size() < 2，这会导致 IndexOutOfBoundsException，但也可能在某些情况下变成 NullPointerException。
        if (graph.getVertices().size() < 2) {
            System.err.println("Error: Not enough vertices in the graph.");
            return;
        }
        //工具栏

        //find 100 nearest vertex
        ToolBar toolBar = new ToolBar();
        //TextField xInput = new TextField();
        //TextField yInput = new TextField();
        TextField pointInput = new TextField();

        Vertex source = graph.getVertices().get(0);
        Vertex destination = graph.getVertices().get(1);

        Button searchButton = new Button("查找最近100个顶点");
        //toolBar.getItems().addAll(new Label("X:"), xInput, new Label("Y:"), yInput, searchButton);
        toolBar.getItems().addAll(new Label("pointID:"), pointInput,  searchButton);


        //处理逻辑

        // TODO: 2025/3/20 输入点数据时才会调用drawmap展示最近100个顶点高亮，在进行其他操作如放大缩小等再次调用drawmap函数时才会展示高亮，因此高亮只会存在一瞬间，进行其他操作之后才会继续产生高亮
        // TODO: 2025/3/22 打算修改成需要产生最短路径和100个顶点高亮时生成那一时刻的静态页面并new一个新的GUI  因为动态车流显示需要时刻绘画边长颜色，如果想要最短路径和顶点高亮一直存在不现实。
        searchButton.setOnAction(e -> {
            try {
                //double x = Double.parseDouble(xInput.getText());
                //double y = Double.parseDouble(yInput.getText());


                int id = Integer.parseInt(pointInput.getText());

                // 查找最近100个顶点
                nearestVertices.set(graph.findNearestVertices(vertexMap,id ,100));
                relatedEdges.set(graph.getRelatedEdges(nearestVertices.get()));

                // 重新绘制地图
                //gc.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());
                //drawMap(gc, graph, source, destination, judgeshortest, nearestVertices.get(), relatedEdges.get());
                redraw(gc, graph, source, destination, judgeshortest, nearestVertices.get(), relatedEdges.get());

            } catch (NumberFormatException ex) {
                System.out.println("请输入有效的数字");
            }
        });

        //放大缩小
        Button zoomInButton = new Button("放大");
        Button zoomOutButton = new Button("缩小");
        Button resetViewButton = new Button("重置视图");
        ComboBox<String> pathOptions = new ComboBox<>();

        //展示最短路径
        pathOptions.getItems().addAll("显示最短路径", "隐藏最短路径");
        pathOptions.setValue("显示最短路径");
        toolBar.getItems().addAll(zoomInButton, zoomOutButton, resetViewButton, pathOptions);
        // 添加事件处理
        zoomInButton.setOnAction(e -> {
            scaleFactor = Math.min(MAX_SCALE, scaleFactor + SCALE_SPEED);
            redraw(gc, graph,source,destination, judgeshortest, nearestVertices.get(), relatedEdges.get());
        });
        zoomOutButton.setOnAction(e -> {
            scaleFactor = Math.max(MIN_SCALE, scaleFactor - SCALE_SPEED);
            redraw(gc, graph,source,destination, judgeshortest, nearestVertices.get(), relatedEdges.get());
        });
        resetViewButton.setOnAction(e -> {
            scaleFactor = 1.0;
            translateX = 0;
            translateY = 0;
            redraw(gc, graph,source,destination, judgeshortest, nearestVertices.get(), relatedEdges.get());
        });
        pathOptions.setOnAction(e -> {
            if (pathOptions.getValue().equals("显示最短路径")) {
                judgeshortest.set(1);
                System.out.println(pathOptions.getValue());
                redraw(gc, graph,source,destination, judgeshortest, nearestVertices.get(), relatedEdges.get());

            } else {
                judgeshortest.set(0);
                System.out.println(pathOptions.getValue());
                redraw(gc, graph,source,destination, judgeshortest, nearestVertices.get(), relatedEdges.get()); // 只重绘地图，不显示路径
            }
        });
        displayShortestPath(gc, graph, source, destination);

        // 绘制地图
        drawMap(gc, graph,source,destination, judgeshortest, nearestVertices.get(), relatedEdges.get());

        double simulationTime=100000;
        double timeStep=10;
        int cars_num=20000;
        TrafficSimulation trafficSimulation = new TrafficSimulation(graph, simulationTime, timeStep,cars_num);
        trafficSimulation.startSimulation();
        new Thread(()->{
            while (true){

                updateMap(gc, trafficSimulation.getGraph(),source,destination, judgeshortest,nearestVertices.get(), relatedEdges.get());

                System.out.println("结束一次车流更新");

                try {
                    Thread.sleep(5000);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            // 更新车流模拟

        }).start();

        // 鼠标事件
        canvas.setOnMousePressed(event -> {
            mousePressedX = event.getSceneX();
            mousePressedY = event.getSceneY();
        });

        canvas.setOnMouseDragged(event -> {
            double deltaX = event.getSceneX() - mousePressedX;
            double deltaY = event.getSceneY() - mousePressedY;

            translateX += deltaX;
            translateY += deltaY;

            // 更新鼠标位置
            mousePressedX = event.getSceneX();
            mousePressedY = event.getSceneY();

            // 重绘地图
            gc.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());
            drawMap(gc, graph,source,destination, judgeshortest, nearestVertices.get(), relatedEdges.get());
            //displayShortestPath(gc, graph, source, destination);
        });

        // 缩放监听事件
        canvas.setOnScroll((ScrollEvent event) -> {
            scaleFactor += (event.getDeltaY() > 0) ? SCALE_SPEED : -SCALE_SPEED;
            scaleFactor = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scaleFactor));


            gc.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());

            drawMap(gc, graph,source,destination, judgeshortest, nearestVertices.get(), relatedEdges.get());
            //displayShortestPath(gc, graph, source, destination);
        });


        VBox root = new VBox(toolBar, canvas);

        // 创建并显示场景
        Scene scene = new Scene(root,1800,1000);
        primaryStage.setTitle("Graph Visualization");
        primaryStage.setScene(scene);
        primaryStage.show();

    }
    private void redraw(GraphicsContext gc, Graph graph,Vertex source,Vertex destination,AtomicInteger judgeshortest,List<Vertex> highlightVertices, List<Edge> highlightEdges) {
        gc.clearRect(0, 0, gc.getCanvas().getWidth(), gc.getCanvas().getHeight());
        drawMap(gc, graph,source,destination,judgeshortest, highlightVertices, highlightEdges);
        // System.out.println(highlightEdges);

    }

    private void drawMap(GraphicsContext gc, Graph graph, Vertex source, Vertex destination,AtomicInteger judgeshortest,List<Vertex> highlightVertices, List<Edge> highlightEdges) {

        //初始化点和边
        graph.getEdges().forEach(edge -> {
            Color edgeColor = getEdgeColor(edge);
            gc.setStroke(edgeColor);

            gc.strokeLine(
                    (edge.start.x + translateX) * scaleFactor, (edge.start.y + translateY) * scaleFactor,
                    (edge.end.x + translateX) * scaleFactor, (edge.end.y + translateY) * scaleFactor
            );
        });

        graph.getVertices().forEach(vertex -> {
            gc.setFill(Color.RED);
            gc.fillOval(
                    (vertex.x + translateX) * scaleFactor - 5,
                    (vertex.y + translateY) * scaleFactor - 5,
                    10, 10
            );
        });

        if(judgeshortest.get() == 1){
            displayShortestPath(gc, graph, source, destination);
            //System.out.println(judgeshortest.get());
        }
        // 高亮最近的100个顶点
       /* gc.setFill(Color.BLUE);
        highlightVertices.forEach(vertex -> {
            gc.fillOval(
                    (vertex.x + translateX) * scaleFactor - 5,
                    (vertex.y + translateY) * scaleFactor - 5,
                    10, 10
            );
        });

        // 高亮相关的边
        gc.setStroke(Color.ORANGE);
        highlightEdges.forEach(edge -> {
            gc.strokeLine(
                    (edge.start.x + translateX) * scaleFactor, (edge.start.y + translateY) * scaleFactor,
                    (edge.end.x + translateX) * scaleFactor, (edge.end.y + translateY) * scaleFactor
            );
        });*/

        displayNearVertex(gc,graph,source,destination,highlightVertices,highlightEdges);
    }
    //展示最近一百个点
    private void displayNearVertex(GraphicsContext gc, Graph graph, Vertex source, Vertex destination,List<Vertex> highlightVertices, List<Edge> highlightEdges){
        // 高亮最近的100个顶点
        gc.setFill(Color.BLUE);
        highlightVertices.forEach(vertex -> {
            gc.fillOval(
                    (vertex.x + translateX) * scaleFactor - 5,
                    (vertex.y + translateY) * scaleFactor - 5,
                    10, 10
            );
        });

        // 高亮相关的边
        gc.setStroke(Color.ORANGE);
        highlightEdges.forEach(edge -> {
            gc.strokeLine(
                    (edge.start.x + translateX) * scaleFactor, (edge.start.y + translateY) * scaleFactor,
                    (edge.end.x + translateX) * scaleFactor, (edge.end.y + translateY) * scaleFactor
            );
        });
        // System.out.println(highlightEdges);
    }

    // TODO: 2025/4/10 设计分段的值的合理性
    private Color getEdgeColor(Edge edge) {
        double traffic = edge.getTrafficTime();
        if (edge.n<0.6*edge.v) {
            return Color.GREEN; // 轻度拥堵
        } else if (0.6*edge.v<=edge.n&&edge.n<=edge.v) {
            return Color.YELLOW; // 中度拥堵
        } else {
            return Color.RED; // 高度拥堵
        }
    }


    private void updateMap(GraphicsContext gc, Graph graph,Vertex source,Vertex destination,AtomicInteger judgeshortest,List<Vertex> highlightVertices, List<Edge> highlightEdges) {
        // 更新并绘制地图
        Platform.runLater(() -> {
            gc.clearRect(0, 0, gc.getCanvas().getWidth(), gc.getCanvas().getHeight());
            drawMap(gc, graph, source, destination, judgeshortest, highlightVertices, highlightEdges);
        });

        graph.getEdges().forEach(edge -> {
            System.out.printf("Edge %d-%d: n=%d, v=%d, trafficTime=%.1f%n",
                    edge.start.id, edge.end.id, edge.n, edge.v, edge.getTrafficTime());
        });

    }

    // TODO: 2025/3/20  展示最优路径，要结合车流量和最短路径综合考虑


    // TODO: 2025/3/20  地图缩放功能只展示重要点的功能。method：可能需要在每个区域set一个特殊点。可能生成连通图的方式需要优化。

    // TODO: 2025/3/21  随着缩放图片或者放大窗口，地图能随着自定义布局。

    // 计算并显示最短路径
    static int outputtimes = 1;
    private void displayShortestPath(GraphicsContext gc, Graph graph, Vertex source, Vertex destination) {

        List<Vertex> path = graph.calculateShortestPath(source, destination);

        gc.setStroke(Color.BLUE);
        for (int i = 0; i < path.size() - 1; i++) {
            Vertex start = path.get(i);
            Vertex end = path.get(i + 1);
            gc.strokeLine(
                    (start.x + translateX) * scaleFactor, (start.y + translateY) * scaleFactor,
                    (end.x + translateX) * scaleFactor, (end.y + translateY) * scaleFactor);
            if(outputtimes>=1){
                System.out.println("start point: (" + start.x * scaleFactor+","+start.y * scaleFactor+")"+
                        "end point: (" + end.x * scaleFactor+","+end.y * scaleFactor+")");}
        }
        outputtimes=0;
    }


}














