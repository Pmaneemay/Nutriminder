import 'package:flutter/material.dart';
import 'package:nutriminder/view/component/custom_appbar.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
     return Scaffold(
      appBar: CustomAppBar(title: ('Home')),
      body: Text('This is HomePage'),
    );
  }
}