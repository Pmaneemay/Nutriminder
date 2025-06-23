import 'package:flutter/material.dart';

class CustomAppBar extends StatelessWidget implements PreferredSizeWidget {
  final String title;
  final bool isChatBotPage;

  const CustomAppBar({super.key, required this.title, this.isChatBotPage = false});

  @override
  Widget build(BuildContext context) {
    return AppBar(
      title: Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
      centerTitle: true,
      actions: [
        IconButton(
          onPressed: isChatBotPage ? null : () {
            Navigator.pushNamed(context, '/chatbot');
          }, 
          icon: const Icon(Icons.message_rounded),
          tooltip: "Chat Bot",
        ),
      ],
    );
  }

  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight);
}
