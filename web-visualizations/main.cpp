//
// Created by Rocky170 on 10/28/2025.
//
#include "tensor_service.h"
#include "websocket_handler.h"
#include "http_handler.h"
#include <iostream>
#include <memory>

int main(int argc, char** argv) {
    const std::string data_file = (argc > 1) ? argv[1] : "./binary-serializer/binary-serializer.bin";
    const int port = (argc > 2) ? std::stoi(argv[2]) : 9001;

    auto tensor_service = std::make_shared<TensorService>(data_file);

    if (!tensor_service->initialize()) {
        std::cerr << "Failed to initialize tensor service\n";
        return 1;
    }

    WebSocketHandler ws_handler(tensor_service);
    HttpHandler http_handler(tensor_service);

    uWS::App app;
    ws_handler.attach_to_app(app);
    http_handler.attach_to_app(app);

    app.listen(port, [port](auto* token) {
        if (token) {
            std::cout << "\n=== Tensor Server Running ===\n";
            std::cout << "Port: " << port << "\n";
            std::cout << "WebSocket: ws://localhost:" << port << "/tensor\n";
            std::cout << "HTTP API:  http://localhost:" << port << "/info\n";
            std::cout << "============================\n\n";
        } else {
            std::cerr << "Failed to bind to port " << port << "\n";
        }
    }).run();

    return 0;
}