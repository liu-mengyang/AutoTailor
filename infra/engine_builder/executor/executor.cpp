#include <stdio.h>
#include <string.h>
#include <chrono>
#include <cctype>
#include <string>
#include <iostream>
#include <unistd.h>
#include <mutex>
#include <thread>

using namespace std;

#define BUF_SIZE 1024

bool battery_is_safe = true;
bool time_is_out = false;
bool is_ret = false;
bool running = true;

string ret_res = "";
mutex m_battery;
mutex m_time;
mutex m_result;
mutex m_ret;

string check_battery_ratio() {
    char buf[BUF_SIZE];
    string result = "";
    float level = 0;
    float scale = 0;
    float ratio = 0;

    shared_ptr<FILE> pipe_battery_level(popen("dumpsys battery | grep level", "r"), pclose);
    if (pipe_battery_level != NULL) {
        while(!feof(pipe_battery_level.get())) {
            if(fgets(buf, sizeof(buf), pipe_battery_level.get()) != NULL) {
                result += buf;
            }
        }
        result.erase(std::remove_if(result.begin(), result.end(), ::isspace), result.end()); // remove spaces
        level = stof(result.substr(result.find(":")+1, result.length()));
    } else {
        printf("popen %s error\n", "dumpsys battery | grep level");
        return "Error";
    }
    result = "";
    shared_ptr<FILE> pipe_battery_scale(popen("dumpsys battery | grep scale", "r"), pclose);
    if (pipe_battery_scale != NULL) {
        while(!feof(pipe_battery_scale.get())) {
            if(fgets(buf, sizeof(buf), pipe_battery_scale.get()) != NULL) {
                result += buf;
            }
        }
        result.erase(std::remove_if(result.begin(), result.end(), ::isspace), result.end()); // remove spaces
        scale = stof(result.substr(result.find(":")+1, result.length()));
    } else {
        printf("popen %s error\n", "dumpsys battery | grep scale");
        return "Error";
    }

    ratio = level / scale;
    cout << "The ratio of battery: " << ratio << endl;

    return to_string(ratio);
}

void battery_safe(float dead_ratio) {
    float begin_ratio = stof(check_battery_ratio());
    while (running) {
        float battery_diff = begin_ratio - stof(check_battery_ratio());
        if (battery_diff >= dead_ratio) {
            lock_guard<mutex> guard(m_battery);
            battery_is_safe = false;
            return;
        }
        sleep(5);
    }
}

void time_out(float ddl) {
    chrono::steady_clock sc;
    auto start = sc.now();
    while (running) {
        sleep(2);
        if ((sc.now() - start)/1ms > ddl) {
            lock_guard<mutex> guard(m_time);
            time_is_out = true;
            return;
        }
    }
}

void execute(const char *cmd) {
    char buf[BUF_SIZE];

    shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);

    if(pipe != NULL) {
        if (!feof(pipe.get())) {
            while(running && fgets(buf, sizeof(buf), pipe.get()) != NULL) {
                lock_guard<mutex> guard(m_result);
                ret_res += buf;
            }
        }
        lock_guard<mutex> guard(m_ret);
        is_ret = true;
        return;
    }
}

string execute_with_ddl(const char *cmd, string result, float ddl) {
    float begin_ratio = stof(check_battery_ratio());

    char buf[BUF_SIZE];

    shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);

    if(pipe != NULL) {
        chrono::steady_clock sc;
        auto start = sc.now();
        auto pt_battery = start;
        bool finish_flag = false;
        bool power_heavy = false;
        int check_frequency = 200; // check battery every 200ms
        // check response under latency constraint
        do {
            if (!feof(pipe.get())) {
                if(fgets(buf, sizeof(buf), pipe.get()) != NULL) {
                    result += buf;
                }
            }
            else {
                finish_flag = true;
                break;
            }
            auto tic = sc.now();
            if ((tic - pt_battery)/1ms >= check_frequency) {

                float battery_diff = begin_ratio - stof(check_battery_ratio());
                power_heavy = battery_diff >= 0.05;
                if (power_heavy) {
                    result = "Notgreen";
                    return result;
                }
                pt_battery = sc.now();
            }
            auto toc = sc.now();
            ddl += (toc - tic)/1ms;
            cout << "Current ddl: " << ddl << endl;

        } while ((sc.now() - start)/1ms <= ddl);

        if(!finish_flag)
            result = "Timeout";

        return result;
    } else {
        printf("popen %s error\n", cmd);
        return "Error";
    }
}

int main(int argc, char *argv[]) {
    using namespace std::literals;

    char* cmd = argv[1];
    float dead_ratio = 0.05;
    float ddl = stof(argv[2]);

    thread t_battery(battery_safe, dead_ratio);
    thread t_time(time_out, ddl);
    thread t_execute(execute, cmd);

    while (1) {
        if (!battery_is_safe) {
            cout << "Notgreen" << endl;
            break;
        }
        if (time_is_out) {
            cout << "Timeout" << endl;
            break;
        }
        if (is_ret) {
            cout << ret_res << endl;
            break;
        }
        sleep(1);
    }
    running = false;

    t_battery.detach();
    t_time.detach();
    t_execute.detach();
    return 0;
}