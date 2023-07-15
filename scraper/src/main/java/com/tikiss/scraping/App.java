package com.tikiss.scraping;

public class App {

    // private static String url =
    // "https://es.wikipedia.org/wiki/Estilometr%C3%ADa";
    private static String url = "https://forocoches.com/foro/showthread.php?t=9077911";
    private static String fileName = ".json";
    private static String folder = "./json/";
    private static int iniPage = 1;
    private static int endPage = 10;
    private static String atrbPage = "page";

    private static int iniThread = 250001;
    private static int endThread = 9999999;

    // private static int endThread = 45000;
    // private static int endThread = 9999999;

    private static int numHilos = 20;
    private static int threadPorHilo = 1000;

    public static void main(String[] args) throws Exception {

        Thread[] pool = new Thread[numHilos];

        for (int i = 0; i < numHilos; i++) {
            pool[i] = new Thread(new Scraper(iniThread, iniThread + threadPorHilo), String.format("SCRAPER-%02d", i));
            pool[i].start();
            iniThread += threadPorHilo;
        }
        int actual = iniThread;
        int t = 0;
        while (actual < endThread) {
            if (!pool[t % numHilos].isAlive()) {
                pool[t % numHilos] = new Thread(new Scraper(iniThread, iniThread + threadPorHilo),
                        String.format("SCRAPER-%02d", t % numHilos));
                pool[t % numHilos].start();
                iniThread += threadPorHilo;
                actual = iniThread;
            }
            t++;
            Thread.sleep(100);
        }

        boolean anyAlive = true;
        while (anyAlive) {
            anyAlive = false;
            for (Thread thread : pool) {
                if (thread.isAlive())
                    anyAlive = true;
                Thread.sleep(100);
            }
        }

        System.out.println("Ultimo hilo " + (iniThread));

        // s.run();

        // s.jsonToCsv("./json/all.json", "./json/all.csv");

    }

}