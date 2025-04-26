import java.util.*;
class Car {
    private Vertex currentVertex; // 当前所在点
    private Vertex destinationVertex; // 前往点
    private double travelTime; // 到达下一个地点耗时t

    public Car(Vertex currentVertex, Graph graph) {
        this.currentVertex = currentVertex;
        this.destinationVertex = null;
        this.travelTime = 0;
        setRandomDestination(currentVertex);
        Edge currentEdge = graph.findEdge(currentVertex, destinationVertex);
        currentEdge.updateTraffic(+1);
        travelTime=currentEdge.getTrafficTime();
    }

    public double traversalTime() {
        return travelTime;
    }

    // 更新车辆状态，包括计时器和路径
    public void update(TrafficSimulation simulation) {
        Graph graph = simulation.getGraph();

        // 记录移动前的原始顶点信息
        Vertex originalStart = currentVertex;
        Vertex originalDestination = destinationVertex;

        // 移动到目的地
        currentVertex = originalDestination;

        // 设置新目的地
        setRandomDestination(currentVertex);

        // 正确查找原路径的边（车辆刚刚离开的边）
        Edge lastEdge = graph.findEdge(originalStart, originalDestination);

        // 查找新路径的边（车辆即将进入的边）
        Edge currentEdge = graph.findEdge(currentVertex, destinationVertex);

        // 更新交通量
        if (lastEdge != null) {
            lastEdge.updateTraffic(-1);  // 正确离开原边
        }
        if (currentEdge != null) {
            currentEdge.updateTraffic(+1);  // 进入新边
            travelTime = currentEdge.getTrafficTime();
        }

    }


    // 设置随机目的地，并计算路径
    public void setRandomDestination(Vertex currentVertex) {
        List<Vertex> possibleDestinations = currentVertex.getConnectedVerticesStream();
        if (possibleDestinations.isEmpty()) return;
        destinationVertex = Vertex.getRandomVertex(possibleDestinations);

    }
}
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


class  TrafficSimulation {

    private final PriorityQueue<TimeSlot> timeSlotsQueue;
    private final Map<Double, TimeSlot> timeSlotsMap; // 模拟中的车辆列表
    private final Graph graph; // 路网图
    private final double simulationTime; // 模拟的总时间（以分钟为单位）
    private double currentTime; // 当前模拟时间（以分钟为单位）
    private final double timeStep; // 模拟时间步长（以分钟为单位）
    private double min_timer;//车队列中最小的timer


    public TrafficSimulation(Graph graph, double simulationTime, double timeStep, int cars_num) {
        this.graph = graph;
        this.simulationTime = simulationTime;
        this.timeStep = timeStep;
        this.currentTime = 0;
        this.min_timer = Double.MAX_VALUE;
        this.timeSlotsQueue = new PriorityQueue<>();
        this.timeSlotsMap = new HashMap<>();
        initializeCars(cars_num);
    }

    private void initializeCars(int cars_num) {
        // 初始化车辆，为每辆车设置起始点和目的地
        for (int i = 0; i < cars_num; i++) { // 假设我们初始化20000辆车
            List<Vertex> possible = graph.getVertices();
            Vertex start = Vertex.getRandomVertex(possible);
            Car car = new Car(start, graph);
            // 计算车辆应属的时间槽起始时间
            double newStartTime = Math.floor((car.traversalTime()+currentTime) / timeStep) * timeStep;

            // 获取或创建新时间槽
            TimeSlot newSlot = timeSlotsMap.get(newStartTime);
            if (newSlot == null) {
                newSlot = new TimeSlot(newStartTime);
                timeSlotsMap.put(newStartTime, newSlot);
                timeSlotsQueue.add(newSlot);
            }

            // 将车辆加入新时间槽
            newSlot.getCars().add(car);
        }
        System.out.println("初始化后有"+timeSlotsMap.size()+"个时间槽");
        assert timeSlotsQueue.peek() != null;
        min_timer = timeSlotsQueue.peek().getStartTime() ;
    }

    public void startSimulation() {
        new Thread(()->{
            while (currentTime < simulationTime) {
                currentTime += timeStep;
                long duration = data.measureTime(() -> {
                    updateSimulation();
                });
                System.out.println("本次车流更新后有 " +timeSlotsQueue.size()  + " 个timeslot");
                System.out.println("本次车流更新耗时: " + duration + " ns");

                //模拟休眠一段时间来模拟现实时间流逝，休眠时间取决于时间步长和模拟速度
                try {
                    Thread.sleep((long) (timeStep*1000 ));
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }


    public Graph getGraph() {
        return graph;
    }

    private void updateSimulation() {

        if (timeSlotsQueue.isEmpty() || currentTime <= min_timer)
            return;

        // 暂存需要重新调度的车辆
        List<Car> rescheduledCars = new ArrayList<>();

        // 从旧时间槽中移除车辆
        for (Car car : timeSlotsQueue.peek().getCars()) {
            rescheduledCars.add(car);
            car.update(this);
        }
        System.out.println("本次更新的车辆数：" + rescheduledCars.size());
        // 如果旧时间槽为空，则清理
        assert timeSlotsQueue.peek() != null;
        timeSlotsMap.remove(timeSlotsQueue.peek().getStartTime());
        timeSlotsQueue.remove(timeSlotsQueue.peek());

        while (!rescheduledCars.isEmpty()) {
            Car car = rescheduledCars.remove(0);
            double newStartTime = Math.floor((car.traversalTime() + currentTime) / timeStep) * timeStep;
            // 获取或创建新时间槽
            TimeSlot newSlot = timeSlotsMap.get(newStartTime);
            if (newSlot == null) {
                newSlot = new TimeSlot(newStartTime);
                timeSlotsMap.put(newStartTime, newSlot);
                timeSlotsQueue.add(newSlot);
            }
            // 将车辆加入新时间槽
            newSlot.getCars().add(car);
            }
            assert timeSlotsQueue.peek() != null;
            min_timer = timeSlotsQueue.peek().getStartTime();
    }


}

