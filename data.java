import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.*;
import javafx.scene.input.ScrollEvent;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

import java.io.*;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;




public class data extends Application {

    protected static double scaleFactor = 4.0;
    private static final double SCALE_SPEED = 0.05;
    private static final double MIN_SCALE = 1;
    private static final double MAX_SCALE = 5.0;
    protected static Canvas canvas = new Canvas(1800, 1000);
    protected static GraphicsContext gc = canvas.getGraphicsContext2D();
    private static double translateX = 0, translateY = 0;  // 用于平移地图的位置
    private double mousePressedX, mousePressedY;
    protected static int printrank;
    protected static volatile AtomicInteger judgeshortest = new AtomicInteger(1);
    protected static volatile AtomicInteger judgebest = new AtomicInteger(1);

    protected static  volatile AtomicReference<List<Vertex>> nearestVertices=new AtomicReference<>(new ArrayList<>());
    protected static volatile AtomicReference<List<Edge>> relatedEdges=new AtomicReference<>(new ArrayList<>()) ;
    protected static volatile AtomicReference<Vertex> source;
    protected static volatile AtomicReference<Vertex> destination;
    @Override
    public void start(Stage primaryStage) {

        int N = 10000;
        double maxCoordinateValue = 1000;
        double maxEdgeLength = 100;
        double connectProbability = 0.2;
        long duration;//计时器
        Graph graph = new Graph();
        File(graph,N,maxCoordinateValue,maxEdgeLength,connectProbability);

        if (graph.getVertices().isEmpty()) {
            System.err.println("Error: Graph has no vertices.");
            return;
        }
        if (graph.getVertices().size() < 2) {
            System.err.println("Error: Not enough vertices in the graph.");
            return;
        }
        //起点终点初始化
        source=new AtomicReference<>(graph.vertices.get(0));
        destination=new AtomicReference<>(graph.vertices.get(1));
        gc.setLineWidth(2);

        //如果 graph.getVertices() 为空或 size() < 2，这会导致 IndexOutOfBoundsException，但也可能在某些情况下变成 NullPointerException。
        if (graph.getVertices().size() < 2) {
            System.err.println("Error: Not enough vertices in the graph.");
            return;
        }
        //工具栏

        //find 100 nearest vertex
        ToolBar toolBar = new ToolBar();
        TextField xInput = new TextField();
        TextField yInput = new TextField();
        TextField startPoint = new TextField("0");
        TextField endPoint = new TextField("1");
        TextField pointInput = new TextField();


        Button searchButton = new Button("查找最近100个顶点");
//        toolBar.getItems().addAll(new Label("pointID:"), pointInput, searchButton);
        toolBar.getItems().addAll(new Label("X:"), xInput, new Label("Y:"), yInput, searchButton);

        //处理逻辑

        // TODO: 2025/3/20 输入点数据时才会调用drawmap展示最近100个顶点高亮，在进行其他操作如放大缩小等再次调用drawmap函数时才会展示高亮，因此高亮只会存在一瞬间，进行其他操作之后才会继续产生高亮
        // TODO: 2025/3/22 打算修改成需要产生最短路径和100个顶点高亮时生成那一时刻的静态页面并new一个新的GUI  因为动态车流显示需要时刻绘画边长颜色，如果想要最短路径和顶点高亮一直存在不现实。
        searchButton.setOnAction(e -> {
            try {
                double x = Double.parseDouble(xInput.getText());
                double y = Double.parseDouble(yInput.getText());
//                int id = Integer.parseInt(pointInput.getText());
                // 查找最近100个顶点
                nearestVertices.set(graph.findNearestVertices(x,y, 100, graph.parts));
                relatedEdges.set(graph.getRelatedEdges(nearestVertices.get()));

                // 重新绘制地图
                scaleFactor = 5;
                translateX = 180-x;
                translateY = 100-y;
                redraw(graph);

            } catch (NumberFormatException ex) {
                System.out.println("请输入有效的数字");
            }
        });

        //放大缩小
        Button zoomInButton = new Button("放大");
        Button zoomOutButton = new Button("缩小");
        Button resetViewButton = new Button("重置视图");
        ComboBox<String> pathOptions1 = new ComboBox<>();
        ComboBox<String> pathOptions2 = new ComboBox<>();
        //起点终点选取

        pathOptions1.getItems().addAll("显示最优路径", "隐藏最优路径");
        pathOptions1.setValue("显示最优路径");

        pathOptions2.getItems().addAll("显示最短路径", "隐藏最短路径");
        pathOptions2.setValue("显示最短路径");

        ToggleGroup toggleGroup = new ToggleGroup();
        ToggleButton setStartPointButton = new ToggleButton("选择起点:");
        setStartPointButton.setToggleGroup(toggleGroup);
        ToggleButton setEndPointButton = new ToggleButton("选择终点:");
        setEndPointButton.setToggleGroup(toggleGroup);



        toolBar.getItems().addAll(zoomInButton, zoomOutButton, resetViewButton, pathOptions1,pathOptions2);
        toolBar.getItems().addAll(setStartPointButton, startPoint, setEndPointButton, endPoint);

        // 添加事件处理
        zoomInButton.setOnAction(e -> {
            scaleFactor = Math.min(MAX_SCALE, scaleFactor + SCALE_SPEED);
            redraw( graph);
        });
        zoomOutButton.setOnAction(e -> {
            scaleFactor = Math.max(MIN_SCALE, scaleFactor - SCALE_SPEED);
            redraw( graph);
        });
        resetViewButton.setOnAction(e -> {
            scaleFactor = 4.0;
            translateX = 0;
            translateY = 0;
            nearestVertices.set(new ArrayList<>());
            relatedEdges.set(new ArrayList<>());
            redraw( graph);
        });


        pathOptions1.setOnAction(e -> {
            // 只重绘地图，不显示路径
            if (pathOptions1.getValue().equals("显示最优路径")) {
                judgebest.set(1);

            } else {
                judgebest.set(0);
            }
            System.out.println(pathOptions1.getValue());

            redraw( graph);
        });

        pathOptions2.setOnAction(e -> {
            // 只重绘地图，不显示路径
            if (pathOptions2.getValue().equals("显示最短路径")) {
                judgeshortest.set(1);

            } else {
                judgeshortest.set(0);
            }
            System.out.println(pathOptions2.getValue());
            redraw( graph);
        });
        double simulationTime=10000000;
        double timeStep=1;
        int cars_num=200000;
        TrafficSimulation trafficSimulation = new TrafficSimulation(graph, simulationTime, timeStep,cars_num);
        // 要测量的函数/代码块
        duration = measureTime(trafficSimulation::startSimulation);
        System.out.println("生成模拟车流耗时: " + duration + " ns");

        // 绘制地图
        duration = measureTime(() -> {
            // 要测量的函数/代码块
            drawMap( graph);
        });
        System.out.println("图像绘制耗时: " + duration + " ns");


        // 鼠标事件
        canvas.setOnMousePressed(event -> {
            mousePressedX = event.getSceneX();
            mousePressedY = event.getSceneY();

            if(setStartPointButton.isSelected())
            {
                double toolbarHeight = toolBar.getHeight();
                int pointID = getPointIDSelected(mousePressedX, mousePressedY, toolbarHeight, gc, graph);
                if(pointID != -1) {
                    startPoint.setText(String.valueOf(pointID));
                    source.set(graph.getVertices().get(getTextFieldPointID(startPoint)));
                }
            }

            if(setEndPointButton.isSelected())
            {
                double toolbarHeight = toolBar.getHeight();
                int pointID = getPointIDSelected(mousePressedX, mousePressedY, toolbarHeight, gc, graph);
                if(pointID != -1) {
                    endPoint.setText(String.valueOf(pointID));
                    destination.set(graph.getVertices().get(getTextFieldPointID(endPoint)));
                }
            }
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
            drawMap( graph);
            //displayShortestPath(gc, graph, source, destination);
        });

        // 缩放监听事件
        canvas.setOnScroll((ScrollEvent event) -> {
            scaleFactor += (event.getDeltaY() > 0) ? SCALE_SPEED : -SCALE_SPEED;
            scaleFactor = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scaleFactor));


            gc.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());

            drawMap( graph);
            //displayShortestPath(gc, graph, source, destination);
        });


        VBox root = new VBox(toolBar, canvas);

        // 创建并显示场景
        Scene scene = new Scene(root, 1800, 1000);
        primaryStage.setTitle("Graph Visualization");
        primaryStage.setScene(scene);
        primaryStage.show();

    }

    private int getPointIDSelected(double mousePressedX, double mousePressedY, double toolbarHeight, GraphicsContext gc, Graph graph)
    {
        double clickRadius = 5;

        double drawX = mousePressedX - clickRadius;  // 修正 X 坐标
        double drawY = mousePressedY - toolbarHeight - clickRadius;  // 修正 Y 坐标

//        gc.setFill(Color.YELLOW);
//        gc.fillOval(drawX, drawY, clickRadius * 2, clickRadius * 2);

        // 遍历所有顶点，检查是否有顶点被圆覆盖
        for (Vertex vertex : graph.getVertices()) {
            if(vertex.rank < printrank) continue;

            double distance = Math.sqrt(Math.pow(drawX - (vertex.x + translateX) * scaleFactor, 2)
                    + Math.pow(drawY - (vertex.y + translateY) * scaleFactor, 2));

            if (distance <= clickRadius * 2) {
                gc.setFill(Color.BLUE);
                gc.fillOval((vertex.x + translateX) * scaleFactor - 5, (vertex.y + translateY) * scaleFactor - 5, 10, 10);
                return vertex.id;
            }
        }
        return -1;
    }

    private int getTextFieldPointID(TextField textField)
    {
        String text = textField.getText().trim();
        if (text.matches("\\d+")) {
            return Integer.parseInt(text);
        }
        return 0;
    }

    private void reflashSource()
    {

    }

    public static void redraw(Graph graph) {
        gc.clearRect(0, 0, gc.getCanvas().getWidth(), gc.getCanvas().getHeight());
        drawMap(graph);
        // System.out.println(highlightEdges);

    }

    private static void drawMap(Graph graph) {

//        if(scaleFactor>=3.5&&scaleFactor<=5)printrank=0;
//        else if(scaleFactor>=1&&scaleFactor<=2)printrank=2;
//        else printrank=3;
        // 分层次显示规则（scaleFactor越小表示视图越放大）
        if (scaleFactor >= 4.0) {
            printrank = 0;  // 最小缩放级别，显示最高层级（最粗略）
        } else if (scaleFactor >= 3.0) {
            printrank = 1;  // 中高缩放级别
        } else if (scaleFactor >= 2.0) {
            printrank = 2;  // 中等缩放级别
        } else {
            printrank = 3;  // 最大缩放级别，显示最低层级（最详细）
        }
        List<Edge> Edges = graph.rankedEdges.getOrDefault(printrank, new ArrayList<>());// 收集合并后的边到新列表
        //System.out.println(" rank:"+printrank+"size:" +Edges.size());
        Edges.forEach(edge -> {
                Color edgeColor = getEdgeColor(edge);
                gc.setStroke(edgeColor);

                gc.strokeLine(
                        (edge.start.x + translateX) * scaleFactor, (edge.start.y + translateY) * scaleFactor,
                        (edge.end.x + translateX) * scaleFactor, (edge.end.y + translateY) * scaleFactor
                );

                // 绘制起点圆点
            // 可根据需要调整基础半径值
            gc.setFill(edgeColor);
                gc.fillOval(
                        (edge.start.x + translateX) * scaleFactor - scaleFactor,
                        (edge.start.y + translateY) * scaleFactor - scaleFactor,
                        2 * scaleFactor,
                        2 * scaleFactor
                );

                // 绘制终点圆点
                gc.fillOval(
                        (edge.end.x + translateX) * scaleFactor - scaleFactor,
                        (edge.end.y + translateY) * scaleFactor - scaleFactor,
                        2 * scaleFactor,
                        2 * scaleFactor
                );

            });

        if (judgebest.get() == 1) {
            displayBestPath(gc, graph, source.get(), destination.get());
            //System.out.println(judgeshortest.get());
        }

        if (judgeshortest.get() == 1) {
            displayShortestPath(gc, graph, source.get(), destination.get());
            //System.out.println(judgeshortest.get());
        }

        displayNearVertex(gc, nearestVertices.get(), relatedEdges.get());
    }


    //展示最近一百个点
    private static void displayNearVertex(GraphicsContext gc, List<Vertex> highlightVertices, List<Edge> highlightEdges) {
        if(highlightVertices.isEmpty()|| highlightEdges.isEmpty())
            return;
        // 高亮最近的100个顶点
        gc.setFill(Color.BLUE);
        highlightVertices.forEach(vertex -> gc.fillOval(
                (vertex.x + translateX) * scaleFactor - scaleFactor,
                (vertex.y + translateY) * scaleFactor - scaleFactor,
                2*scaleFactor, 2*scaleFactor
        ));

        // 高亮相关的边
        gc.setStroke(Color.ORANGE);
        highlightEdges.forEach(edge -> gc.strokeLine(
                (edge.start.x + translateX) * scaleFactor, (edge.start.y + translateY) * scaleFactor,
                (edge.end.x + translateX) * scaleFactor, (edge.end.y + translateY) * scaleFactor
        ));
        // System.out.println(highlightEdges);
    }

    private static Color getEdgeColor(Edge edge) {
        if(edge.v==0) return Color.GRAY;//缩略线
        if (edge.n<0.6*edge.v) {
            return Color.GREEN; // 轻度拥堵
        } else if (0.6*edge.v<=edge.n&&edge.n<=edge.v) {
            return Color.YELLOW; // 中度拥堵
        } else {
            return Color.RED; // 高度拥堵
        }
    }

    // TODO: 2025/3/20  展示最优路径，要结合车流量和最短路径综合考虑


    // TODO: 2025/3/20  地图缩放功能只展示重要点的功能。method：可能需要在每个区域set一个特殊点。可能生成连通图的方式需要优化。

    // TODO: 2025/3/21  随着缩放图片或者放大窗口，地图能随着自定义布局。

    // 计算并显示最短路径


    private static void displayBestPath(GraphicsContext gc, Graph graph, Vertex source, Vertex destination) {

        List<Vertex> path = graph.calculateBestPath(source, destination);

        gc.setStroke(Color.BLUE);
        for (int i = 0; i < path.size() - 1; i++) {
            Vertex start = path.get(i);
            Vertex end = path.get(i + 1);
            gc.strokeLine(
                    (start.x + translateX) * scaleFactor, (start.y + translateY) * scaleFactor,
                    (end.x + translateX) * scaleFactor, (end.y + translateY) * scaleFactor);

        }
    }
    private static void displayShortestPath(GraphicsContext gc, Graph graph, Vertex source, Vertex destination) {

        List<Vertex> path = graph.calculateShortestPath(source, destination);

        gc.setStroke(Color.BLUE);
        for (int i = 0; i < path.size() - 1; i++) {
            Vertex start = path.get(i);
            Vertex end = path.get(i + 1);
            gc.strokeLine(
                    (start.x + translateX) * scaleFactor, (start.y + translateY) * scaleFactor,
                    (end.x + translateX) * scaleFactor, (end.y + translateY) * scaleFactor);

        }

    }
    private static void File(Graph graph,int N,double maxCoordinateValue,double maxEdgeLength,double connectProbability) {
        String filePath = System.getProperty("user.dir") + "\\location.txt";
        File file = new File(filePath);
        // 检查文件是否存在
        if (!file.exists()) {
            try {
                // 如果文件不存在，则创建指定名称的新文件
                if (file.createNewFile()) {
                    System.out.println("文件已创建: " + file.getName());
                } else {
                    System.out.println("文件创建失败！");
                }
            } catch (IOException e) {
                System.out.println("发生错误：" + e.getMessage());
                e.printStackTrace();
            }
        } else {
            System.out.println("文件已存在: " + file.getName());
        }
        //文件存在且为空，随机生成并写入
        if (file.exists() && file.length() == 0) {
            try {
                BufferedWriter writer = new BufferedWriter(new FileWriter(file, true));
                graph.generateConnectedGraph(N,maxEdgeLength, connectProbability,maxCoordinateValue);

                //写入点
                writer.write("Vertex:");
                writer.newLine();
                for (int i = 0; i < N; i++) {
                    writer.write(i + " " +graph.vertices.get(i).x + " " +graph.vertices.get(i).y);
                    writer.newLine();  // 写入换行符
                }

                //写入边
                writer.write("Edge:");
                writer.newLine();
                Set<String> set = new HashSet<>();
                //去重
                for (Vertex v:graph.vertices) {
                    for(Edge e:v.edges[0]) {
                        set.add(e.start.id + " " + e.end.id+" "+e.v);
                    }
                }
                for(String s:set) {
                    writer.write(s);
                    writer.newLine();
                }

                // 关闭 BufferedWriter
                writer.close();
            } catch (IOException e) {
                e.fillInStackTrace();
            }
        } else {
            //文件存在不为空开始读文件
            try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
                graph.parts = new Partition(maxEdgeLength);
                boolean isVertexSection = false;
                boolean isEdgeSection = false;
                String line;

                while ((line = reader.readLine()) != null) {
                    if (line.startsWith("Vertex:")) {
                        isVertexSection = true;  // 标记已开始解析顶点数据
                        continue;
                    }
                    if (line.startsWith("Edge:")) {
                        isVertexSection = false;  // 标记已开始解析顶点数据
                        isEdgeSection = true;
                        continue;
                    }

                    //读点
                    if (isVertexSection && !line.isEmpty()) {
                        String[] s = line.split("\\s+");
                        if (s.length == 3) {
                            try {
                                int id = Integer.parseInt(s[0]);
                                double x = Double.parseDouble(s[1]);
                                double y = Double.parseDouble(s[2]);
                                Vertex vertex = new Vertex(id, x, y);
                                graph.parts.addVertex(vertex);
                                graph.vertices.add(vertex);
                            } catch (NumberFormatException e) {
                                System.err.println("无效的顶点数据: " + line);
                            }
                        }
                    }

                    //读边
                    if (isEdgeSection && !line.isEmpty()) {
                        String[] s = line.split("\\s+");
                        if (s.length == 3) {
                            try {
                                int startId = Integer.parseInt(s[0]);
                                int endId = Integer.parseInt(s[1]);
                                int container = Integer.parseInt(s[2]);
                            } catch (NumberFormatException e) {
                                System.err.println("无效的边数据: " + line);
                            }
                            Edge edge = new Edge(graph.vertices.get(Integer.parseInt(s[0])), graph.vertices.get(Integer.parseInt(s[1])), Integer.parseInt(s[2]));
                            graph.parts.addEdge(edge);
                            edge.start.edges[0].add(edge);
                            edge.end.edges[0].add(edge);
                        }
                    }

                }
            } catch (IOException e) {
                e.fillInStackTrace();
            }
            graph.generateRankedEdges(15,5);
        }
    }

    public static long measureTime(Runnable task) {
        long start = System.nanoTime();
        task.run();
        return System.nanoTime() - start;
    }
}













