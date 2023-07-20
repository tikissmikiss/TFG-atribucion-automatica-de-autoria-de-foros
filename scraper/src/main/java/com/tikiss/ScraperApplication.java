package com.tikiss;

import com.tikiss.scraping.Scraper;
import org.apache.commons.cli.*;

public class ScraperApplication {
    private static final String url = "https://forocoches.com/foro/showthread.php?t=9077911";
    private static String fileName = ".json";
    private static String folder = "./json/";
    private static int iniPage = 1;
    private static int endPage = 10;
    private static String atrbPage = "page";
    private static int iniThread = 1;
    private static int endThread = 9999999;
    private static int numHilos = 20;
    private static int threadPorHilo = 1000;

    public static void main(String[] args) throws Exception {
        Options options = new Options();
        options.addOption("f", "file", true, "Nombre del archivo de salida");
        options.addOption("d", "directory", true, "Directorio de salida");
        options.addOption("ip", "initpage", true, "Página de inicio. Valor inicial del atributo de la url.");
        options.addOption("ep", "endpage", true, "Página final. Valor final del atributo de la url.");
        options.addOption("ap", "atrpage", true, "Atributo de página de la url. p. e. page='pag' en https://foro.com/showthread.php?t=9077911&pag=1");
        options.addOption("ih", "iniHilo", true, "Hilo del foro de inicio. Valor inicial del atributo de la url. p.e. t=1");
        options.addOption("eh", "endHilo", true, "Hilo del foro final. Valor final del atributo de la url. p.e. t=90.");
        options.addOption("t", "numThread", true, "Número de hilos de ejecución. Cada hilo ejecuta un número de hilos de scraping.");
        options.addOption("h", "help", false, "Mostrar ayuda");

        CommandLineParser parser = new DefaultParser();
        try {
            CommandLine cmd = parser.parse(options, args);

            if (cmd.hasOption("h")) {
                printHelp(options);
                System.exit(0);
            }

            setOptions(cmd);

            executeScraperThreads();

        } catch (ParseException e) {
            System.err.println("Error al analizar los argumentos: " + e.getMessage());
            System.exit(1);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void printHelp(Options options) {
        HelpFormatter formatter = new HelpFormatter();
        formatter.setSyntaxPrefix("Uso: ");
        formatter.printHelp("java -jar web-scraper-0.1.0.jar [opciones]\n\n", options);
    }

    private static void setOptions(CommandLine cmd) {
        if (cmd.hasOption("f")) {
            fileName = cmd.getOptionValue("f");
        }
        if (cmd.hasOption("d")) {
            folder = cmd.getOptionValue("d");
        }
        if (cmd.hasOption("ip")) {
            iniPage = Integer.parseInt(cmd.getOptionValue("ip"));
        }
        if (cmd.hasOption("ep")) {
            endPage = Integer.parseInt(cmd.getOptionValue("ep"));
        }
        if (cmd.hasOption("ap")) {
            atrbPage = cmd.getOptionValue("ap");
        }
        if (cmd.hasOption("ih")) {
            iniThread = Integer.parseInt(cmd.getOptionValue("ih"));
        }
        if (cmd.hasOption("eh")) {
            endThread = Integer.parseInt(cmd.getOptionValue("eh"));
        }
        if (cmd.hasOption("t")) {
            numHilos = Integer.parseInt(cmd.getOptionValue("t"));
        }
        if (cmd.hasOption("tph")) {
            threadPorHilo = Integer.parseInt(cmd.getOptionValue("tph"));
        }
    }

    private static void executeScraperThreads() throws InterruptedException {
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
                pool[t % numHilos] = new Thread(new Scraper(iniThread, iniThread + threadPorHilo), String.format("SCRAPER-%02d", t % numHilos));
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
                if (thread.isAlive()) anyAlive = true;
                Thread.sleep(100);
            }
        }

        System.out.println("Último hilo: " + iniThread);
    }
}
