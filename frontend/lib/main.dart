import 'package:flutter/material.dart';
import 'package:nutriminder/view/chatbot_page.dart';
import 'package:nutriminder/view/home_page.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:nutriminder/view/login_page.dart';

final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();
void main() async {
  WidgetsFlutterBinding.ensureInitialized(); 
  await dotenv.load(fileName: ".env");
  runApp( MaterialApp(
     title: 'Nutriminder',
     debugShowCheckedModeBanner: false,
     navigatorKey: navigatorKey,
     initialRoute: '/',
     routes: {
      '/' : (context) => const HomePage(),
      '/chatbot' : (context) => const ChatbotPage(),
      '/login' : (context) => const LoginPage(),
     },
    ));
}

