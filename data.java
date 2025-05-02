import javafx.application.Application;
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




public class data extends Application {

    private double scaleFactor = 4.0;
    private static final double SCALE_SPEED = 0.05;
    private static final double MIN_SCALE = 1;
    private static final double MAX_SCALE = 5.0;

    private double translateX = 0, translateY = 0;  // 用于平移地图的位置
    private double mousePressedX, mousePressedY;

    AtomicInteger judgeshortest = new AtomicInteger(1);
    AtomicInteger judgebest = new AtomicInteger(1);

    AtomicReference<List<Vertex>> nearestVertices = new AtomicReference<>(new ArrayList<>());
    AtomicReference<List<Edge>> relatedEdges = new AtomicReference<>(new ArrayList<>());

    @Override
    public void start(Stage primaryStage) {

        int N = 10000;
        double maxCoordinateValue = 1000;
        double maxEdgeLength = 100;
        double connectProbability = 0.2;
        Graph graph = new Graph();
        long duration = measureTime(() -> graph.generateConnectedGraph(N,maxEdgeLength, connectProbability,maxCoordinateValue));
        System.out.println("图生成耗时: " + duration + " ns");


        if (graph.getVertices().isEmpty()) {
            System.err.println("Error: Graph has no vertices.");
            return;
        }
        if (graph.getVertices().size() < 2) {
            System.err.println("Error: Not enough vertices in the graph.");
            return;
        }

        Canvas canvas = new Canvas(1800, 1000);
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
        TextField xInput = new TextField();
        TextField yInput = new TextField();
        TextField startPoint = new TextField("0");
        TextField endPoint = new TextField("1");
        TextField pointInput = new TextField();
        AtomicReference<Vertex> source = new AtomicReference<>(graph.getVertices().get(getTextFieldPointID(startPoint)));
        AtomicReference<Vertex> destination = new AtomicReference<>(graph.getVertices().get(getTextFieldPointID(endPoint)));

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
                redraw(scaleFactor,gc, graph, source.get(), destination.get(),judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get());

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
            redraw(scaleFactor,gc, graph, source.get(), destination.get(),judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get());
        });
        zoomOutButton.setOnAction(e -> {
            scaleFactor = Math.max(MIN_SCALE, scaleFactor - SCALE_SPEED);
            redraw(scaleFactor,gc, graph, source.get(), destination.get(),judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get());
        });
        resetViewButton.setOnAction(e -> {
            scaleFactor = 4.0;
            translateX = 0;
            translateY = 0;
            redraw(scaleFactor,gc, graph, source.get(), destination.get(),judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get());
        });


        pathOptions1.setOnAction(e -> {
            // 只重绘地图，不显示路径
            if (pathOptions1.getValue().equals("显示最优路径")) {
                judgebest.set(1);

            } else {
                judgebest.set(0);
            }
            System.out.println(pathOptions1.getValue());
            redraw(scaleFactor,gc, graph, source.get(), destination.get(),judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get());
        });

        pathOptions2.setOnAction(e -> {
            // 只重绘地图，不显示路径
            if (pathOptions2.getValue().equals("显示最短路径")) {
                judgeshortest.set(1);

            } else {
                judgeshortest.set(0);
            }
            System.out.println(pathOptions2.getValue());
            redraw(scaleFactor,gc, graph, source.get(), destination.get(),judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get());
        });

        // 绘制地图
        duration = measureTime(() -> {
            // 要测量的函数/代码块
            drawMap(scaleFactor,gc, graph, source.get(), destination.get(), judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get());
        });
        System.out.println("图像绘制耗时: " + duration + " ns");


        double simulationTime=100000;
        double timeStep=1;
        int cars_num=200000;
        TrafficSimulation trafficSimulation = new TrafficSimulation(graph, simulationTime, timeStep,cars_num);

        // 要测量的函数/代码块
        duration = measureTime(trafficSimulation::startSimulation);
        System.out.println("生成模拟车流耗时: " + duration + " ns");

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
            drawMap(scaleFactor,gc, graph, source.get(), destination.get(),judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get());
            //displayShortestPath(gc, graph, source, destination);
        });

        // 缩放监听事件
        canvas.setOnScroll((ScrollEvent event) -> {
            scaleFactor += (event.getDeltaY() > 0) ? SCALE_SPEED : -SCALE_SPEED;
            scaleFactor = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scaleFactor));


            gc.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());

            drawMap(scaleFactor,gc, graph, source.get(), destination.get(),judgebest, judgeshortest, nearestVertices.get(), relatedEdges.get());
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

    private void redraw(double scaleFactor,GraphicsContext gc, Graph graph, Vertex source, Vertex destination,AtomicInteger judgeBest, AtomicInteger judgeshortest, List<Vertex> highlightVertices, List<Edge> highlightEdges) {
        gc.clearRect(0, 0, gc.getCanvas().getWidth(), gc.getCanvas().getHeight());
        drawMap(scaleFactor,gc, graph, source, destination,  judgeBest,judgeshortest, highlightVertices, highlightEdges);
        // System.out.println(highlightEdges);

    }

    int printrank;
    private void drawMap(double scaleFactor,GraphicsContext gc, Graph graph, Vertex source, Vertex destination,AtomicInteger judgeBest, AtomicInteger judgeshortest, List<Vertex> highlightVertices, List<Edge> highlightEdges) {

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

        if (judgeBest.get() == 1) {
            displayBestPath(gc, graph, source, destination);
            //System.out.println(judgeshortest.get());
        }

        if (judgeshortest.get() == 1) {
            displayShortestPath(gc, graph, source, destination);
            //System.out.println(judgeshortest.get());
        }
        displayNearVertex(gc,  highlightVertices, highlightEdges);
    }


    //展示最近一百个点
    private void displayNearVertex(GraphicsContext gc, List<Vertex> highlightVertices, List<Edge> highlightEdges) {
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

    private Color getEdgeColor(Edge edge) {
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


    private void displayBestPath(GraphicsContext gc, Graph graph, Vertex source, Vertex destination) {

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
    private void displayShortestPath(GraphicsContext gc, Graph graph, Vertex source, Vertex destination) {

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

    public static long measureTime(Runnable task) {
        long start = System.nanoTime();
        task.run();
        return System.nanoTime() - start;
    }
}













