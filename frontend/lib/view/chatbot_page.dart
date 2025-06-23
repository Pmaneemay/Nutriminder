import 'package:flutter/material.dart';
import 'package:nutriminder/view/component/custom_appbar.dart';

class ChatbotPage extends StatelessWidget {
  const ChatbotPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: CustomAppBar(title: ('Nutriminder ChatBot'), isChatBotPage: true,),
      body: Text('This is chatbot page'),
    );
  }
}