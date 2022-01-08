import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

public class Main {

    private static int N, M, n, m, p;
    private static int[][] image, filter;
    private static CyclicBarrier barrier;

    // Calculează valoarea nouă a pixelului de pe poziția i,j
    public static int fct(int i, int j){

        int result = 0;
        int offset = n/2;
        for(int ii = 0; ii < n; ii ++){
            for(int jj = 0; jj < m; jj ++){
                result += filter[ii][jj] * image[i-offset+ii][j-offset+jj];
            }
        }
        return result;
    }

    public static class MyThread extends Thread {

        // Definirea clasei proprii de thread-uri
        // Atribute: step si start

        int step, start;

        MyThread(int start, int step) {
            this.step = step;
            this.start = start;
        }

        public void run() {
            // Calcularea matricei filtrate folosind metoda saririi peste un numar step de elemente

            int currentIndex = this.start;
            List<Integer> results = new ArrayList<>();

            while (currentIndex < N * M) {
                results.add(fct(currentIndex / M + 2, currentIndex % M + 2));
                currentIndex += this.step;
            }

            currentIndex = currentIndex % p;
            int index = 0;
            try {
                barrier.await();
            } catch (InterruptedException | BrokenBarrierException e) {
                e.printStackTrace();
            }

            while(currentIndex < N*M){
                image[currentIndex / M + 2][currentIndex % M + 2] = results.get(index++);
                currentIndex += this.step;
            }

        }
    }

    public static void thread() throws InterruptedException, IOException {
        // Definim threadurile

        Thread[] threads = new MyThread[p];
        FileWriter writer = new FileWriter(new File("parallelResult.txt"));
        for(int i = 0; i < p; i ++){
            threads[i] = new MyThread(i, p);
        }

        // Pornim threadurile si apoi le facem join

        for(int i = 0; i < p; i ++)
            threads[i].start();

        for(int i = 0; i < p; i ++)
            threads[i].join();
        
        for(int i = 2; i < N + 2; i ++)
            for(int j = 2; j < M + 2; j ++)
                writer.write(image[i][j] + " ");
        writer.close();

    }


    public static void sequential() throws IOException {

        FileWriter writer = new FileWriter(new File("sequentialResult.txt"));
        List<Integer> results = new ArrayList<>();
        for(int i = 2; i < N+2; i ++){
            for(int j = 2; j < M+2; j ++) {
                results.add(fct(i, j));
            }
        }
        int index = 0;
        for(int i = 2; i < N+2; i ++){
            for(int j = 2; j < M+2; j ++, index++){
                image[i][j] = results.get(index);
            }
        }

        for(int i = 2; i < N+2; i ++){
            for(int j = 2; j < M+2; j ++){
                writer.write(image[i][j] + " ");
            }
        }
        writer.close();

    }

    public static void initData() throws IOException {
        barrier = new CyclicBarrier(p);
        image = new int[N + 5][M + 5];
        filter = new int[n][m];

        File imageFile = new File("imageMatrix.txt");
        File filterFile = new File("filterMatrix.txt");

        // Citirea matricei imaginii din fisier

        Scanner fileReader = new Scanner(imageFile);
        for (int i = 0; i < N; i++){
            for (int j = 0; j < M; j++) {
                image[i + 2][j + 2] = fileReader.nextInt();
            }
        }
        fileReader.close();

        // Bordarea matricii care reprezinta imaginea cu 2 linii si coloane ( in cazul in care matricea de filtru este 5x5 )

        image[0][0] = image[0][1] = image[1][0] = image[1][1] = image[2][2];
        image[N+3][1] = image[N+2][1] = image[N+3][0] = image[N+2][0] = image[N+1][2];
        image[0][M+3] = image[0][M+2] = image[1][M+3] = image[1][M+2] = image[2][M+1];
        image[N+3][M+3] = image[N+3][M+2] = image[N+2][M+3] = image[N+2][M+2] = image[N+1][M+1];

        for(int i = 2; i < N+2; i ++) {
            image[i][0] = image[i][1] = image[i][2];
            image[i][M+2] = image[i][M+3] = image[i][M+1];
        }
        for(int j = 2; j < M+2; j ++) {
            image[0][j] = image[1][j] = image[2][j];
            image[N + 2][j] = image[N + 3][j] = image[N + 1][j];
        }

        // Citirea matricei filtru din fisier

        fileReader = new Scanner(filterFile);
        for (int i = 0; i < n; i++){
            for (int j = 0; j < m; j++) {
                filter[i][j] = fileReader.nextInt();
            }
        }
        fileReader.close();

        // Rularea algoritmului de filtrare si obtinerea timpului de executie
        File seqResult = new File("sequentialResult.txt");
        File parResult = new File("parallelResult.txt");
        seqResult.createNewFile();
        parResult.createNewFile();
    }

    public static void main(String[] args) {

        try {
            // Initializarea datelor

            N = Integer.parseInt(args[0]);
            M = Integer.parseInt(args[1]);
            n = m = Integer.parseInt(args[2]);
            p = Integer.parseInt(args[3]);

            // Crearea fisierelor

            if(args[0].equals("create")) {
                Utils.createFile("imageMatrix.txt", N * M, 0, 10);
                Utils.createFile("filterMatrix.txt", n * m, -5, 5);
                System.out.println("Files created!");
                return;
            }

            // Sequential
            initData();
            long startTime = System.nanoTime();
            sequential();
            long finishTime = System.nanoTime();
            System.out.println((double)(finishTime - startTime));

            // Parallel
            initData();
            startTime = System.nanoTime();
            thread();
            finishTime = System.nanoTime();
            System.out.println((double)(finishTime - startTime));

            // Verificam daca rezultatele obtinute sunt identice
            System.out.println("Checking results...");
            if(!Utils.checkEquality("sequentialResult.txt", "parallelResult.txt", "INT"))
                System.out.println("Bad results!!!");
            else System.out.println("The results are identical!");
        }
        catch(Exception exception){
            exception.printStackTrace();
        }
    }

}